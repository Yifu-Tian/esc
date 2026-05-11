from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mm_flow.three_d.kinematics import wrap_to_pi


_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_MOMA_ASSET_ROOT = _PACKAGE_ROOT / "assets" / "robots" / "moma" / "whole_body_description"


@dataclass(frozen=True)
class _JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class _VisualSpec:
    link: str
    geometry_type: str
    origin_matrix: np.ndarray
    mesh_path: Path | None = None
    box_size: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _CollisionProxySpec:
    link: str
    proxy_type: str
    origin_matrix: np.ndarray
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None


@dataclass(frozen=True)
class MomaPiperMobileManipulator3D:
    """URDF-derived kinematic scaffold for the mobile Piper platform.

    State is [x, y, yaw, joint1, ..., joint6]. The base uses the 9-DoF URDF's
    virtual x/y/yaw joints, while collision checking still uses a conservative
    capsule approximation for the arm links.
    """

    urdf_path: Path = _MOMA_ASSET_ROOT / "urdf" / "mobile_piper_9dof.urdf"
    mesh_root: Path = _MOMA_ASSET_ROOT / "meshes"
    base_radius: float = 0.42
    base_height: float = 0.34
    link_radius: float = 0.055

    def __post_init__(self) -> None:
        urdf_path = Path(self.urdf_path)
        all_joints = _load_joints(urdf_path)
        arm_joints = tuple(joint for joint in all_joints if joint.name in {f"joint{i}" for i in range(1, 7)})
        object.__setattr__(self, "_all_joints", all_joints)
        object.__setattr__(self, "_arm_joints", arm_joints)
        object.__setattr__(self, "q_limits", tuple((j.lower, j.upper) for j in self._arm_joints))
        object.__setattr__(self, "_visual_specs", _load_visual_specs(urdf_path, Path(self.mesh_root)))
        object.__setattr__(self, "_collision_proxy_specs", _build_visual_bbox_collision_proxies(self._visual_specs))

    @property
    def state_dim(self) -> int:
        return 3 + len(self._arm_joints)

    @property
    def mount_height(self) -> float:
        return 0.435

    @property
    def max_reach(self) -> float:
        return 0.82

    def forward_kinematics(self, state: np.ndarray) -> dict[str, np.ndarray]:
        state = np.asarray(state, dtype=float)
        x, y, yaw = state[:3]
        q = self.clip_joints(state[3:])

        base = np.array([x, y, 0.0], dtype=float)
        base_center = np.array([x, y, 0.5 * self.base_height], dtype=float)
        link_transforms = self.link_transforms(state)
        chain_links = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base"]
        points = [link_transforms[name][:3, 3].copy() for name in chain_links if name in link_transforms]
        ee = link_transforms.get("gripper_base", link_transforms["link6"])[:3, 3].copy()
        segments = np.array([
            [a[0], a[1], a[2], b[0], b[1], b[2]]
            for a, b in zip(points[:-1], points[1:])
            if np.linalg.norm(b - a) > 1e-6
        ])
        if len(segments) == 0:
            segments = np.zeros((0, 6), dtype=float)

        return {
            "base": base,
            "base_center": base_center,
            "mount": points[0],
            "elbow": points[min(3, len(points) - 1)],
            "ee": ee,
            "points": np.array(points),
            "segments": segments,
            "link_transforms": link_transforms,
        }

    def link_transforms(self, state: np.ndarray) -> dict[str, np.ndarray]:
        state = np.asarray(state, dtype=float)
        x, y, yaw = state[:3]
        q = self.clip_joints(state[3:])
        q_by_name = {f"joint{i}": q[i - 1] for i in range(1, 7)}

        transforms = {
            "mobile_base": _transform_xyz_rpy(
                np.array([x, y, 0.0], dtype=float),
                np.array([0.0, 0.0, yaw], dtype=float),
            )
        }
        pending = list(self._all_joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint.parent not in transforms:
                    continue
                t = transforms[joint.parent] @ _transform_xyz_rpy(joint.xyz, joint.rpy)
                if joint.name in q_by_name and joint.joint_type == "revolute":
                    t = t @ _rotation_about_axis(joint.axis, q_by_name[joint.name])
                transforms[joint.child] = t
                pending.remove(joint)
                progressed = True
            if not progressed:
                break
        return transforms

    def end_effector(self, state: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(state)["ee"]

    def inverse_kinematics_for_goal(
        self,
        base_xy: np.ndarray,
        yaw: float,
        goal_xyz: np.ndarray,
        elbow_up: bool = False,
    ) -> np.ndarray:
        q0 = _seed_joint_configuration(elbow_up)

        def residual(q: np.ndarray) -> np.ndarray:
            state = np.array([base_xy[0], base_xy[1], yaw, *q], dtype=float)
            ee_error = self.end_effector(state) - goal_xyz
            regularization = 0.015 * (q - q0)
            return np.concatenate([ee_error, regularization])

        lo = np.array([lim[0] for lim in self.q_limits], dtype=float)
        hi = np.array([lim[1] for lim in self.q_limits], dtype=float)
        result = least_squares(
            residual,
            np.clip(q0, lo, hi),
            bounds=(lo, hi),
            max_nfev=80,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
        )
        return self.clip_joints(result.x)

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        lo = np.array([lim[0] for lim in self.q_limits], dtype=float)
        hi = np.array([lim[1] for lim in self.q_limits], dtype=float)
        clipped = np.clip(q, lo, hi)
        return np.array([wrap_to_pi(v) if lo_i < -3.0 and hi_i > 3.0 else v for v, lo_i, hi_i in zip(clipped, lo, hi)])

    def export_visual_assets(self, output_dir: Path) -> list[dict]:
        asset_dir = output_dir / "moma_meshes"
        asset_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        copied: dict[Path, str] = {}
        for spec in self._visual_specs:
            item = {
                "link": spec.link,
                "type": spec.geometry_type,
                "originMatrix": spec.origin_matrix.tolist(),
            }
            if spec.geometry_type == "mesh" and spec.mesh_path is not None:
                source = spec.mesh_path.resolve()
                if source not in copied:
                    target = asset_dir / source.name
                    shutil.copy2(source, target)
                    copied[source] = f"moma_meshes/{target.name}"
                item["url"] = copied[source]
                item["format"] = source.suffix.lower().lstrip(".")
            elif spec.geometry_type == "box" and spec.box_size is not None:
                item["size"] = list(spec.box_size)
            exported.append(item)
        return exported

    def export_collision_proxies(self) -> list[dict]:
        proxies = []
        for spec in self._collision_proxy_specs:
            item = {
                "link": spec.link,
                "type": spec.proxy_type,
                "originMatrix": spec.origin_matrix.tolist(),
            }
            if spec.size is not None:
                item["size"] = list(spec.size)
            if spec.radius is not None:
                item["radius"] = spec.radius
            if spec.length is not None:
                item["length"] = spec.length
            proxies.append(item)
        return proxies


def _load_joints(urdf_path: Path) -> tuple[_JointSpec, ...]:
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name.startswith("virtual_joint"):
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        xyz = _parse_vec(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _parse_vec(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        axis_xyz = _parse_vec(axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1")
        lower = float(limit.attrib.get("lower", "-3.14159") if limit is not None else -np.pi)
        upper = float(limit.attrib.get("upper", "3.14159") if limit is not None else np.pi)
        joints.append(
            _JointSpec(
                name=name,
                joint_type=joint.attrib["type"],
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=xyz,
                rpy=rpy,
                axis=axis_xyz,
                lower=lower,
                upper=upper,
            )
        )
    if len([joint for joint in joints if joint.name in {f"joint{i}" for i in range(1, 7)}]) != 6:
        raise ValueError(f"Expected 6 Piper arm joints in {urdf_path}")
    return tuple(joints)


def _load_visual_specs(urdf_path: Path, mesh_root: Path) -> tuple[_VisualSpec, ...]:
    root = ET.parse(urdf_path).getroot()
    visuals = []
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        visual = link.find("visual")
        if visual is None:
            continue
        origin = visual.find("origin")
        xyz = _parse_vec(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _parse_vec(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh = geometry.find("mesh")
        box = geometry.find("box")
        if mesh is not None:
            visuals.append(
                _VisualSpec(
                    link=link_name,
                    geometry_type="mesh",
                    origin_matrix=_transform_xyz_rpy(xyz, rpy),
                    mesh_path=_resolve_mesh_path(mesh.attrib["filename"], mesh_root),
                )
            )
        elif box is not None:
            visuals.append(
                _VisualSpec(
                    link=link_name,
                    geometry_type="box",
                    origin_matrix=_transform_xyz_rpy(xyz, rpy),
                    box_size=tuple(float(v) for v in box.attrib["size"].split()),
                )
            )
    return tuple(visuals)


def _resolve_mesh_path(filename: str, mesh_root: Path) -> Path:
    basename = Path(filename).name
    candidates = [
        mesh_root / basename,
        mesh_root / "dae" / f"{Path(basename).stem}.dae",
        mesh_root / f"{Path(basename).stem}.dae",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve mesh {filename!r} under {mesh_root}")


def _build_visual_bbox_collision_proxies(visual_specs: tuple[_VisualSpec, ...]) -> tuple[_CollisionProxySpec, ...]:
    proxies = []
    for spec in visual_specs:
        if spec.geometry_type == "box" and spec.box_size is not None:
            proxies.append(
                _CollisionProxySpec(
                    link=spec.link,
                    proxy_type="box",
                    origin_matrix=spec.origin_matrix,
                    size=tuple(float(v) for v in spec.box_size),
                )
            )
        elif spec.geometry_type == "mesh" and spec.mesh_path is not None:
            center, size = _mesh_aabb_after_origin(spec.mesh_path, spec.origin_matrix)
            proxies.append(
                _CollisionProxySpec(
                    link=spec.link,
                    proxy_type="box",
                    origin_matrix=_transform_xyz_rpy(center, np.zeros(3)),
                    size=tuple(float(max(v, 1e-3)) for v in size),
                )
            )
    return tuple(proxies)


def _mesh_aabb_after_origin(mesh_path: Path, origin_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = _load_mesh_vertices(mesh_path)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    transformed = (origin_matrix @ homogeneous.T).T[:, :3]
    lower = transformed.min(axis=0)
    upper = transformed.max(axis=0)
    return 0.5 * (lower + upper), upper - lower


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        return _load_binary_stl_vertices(mesh_path)
    if suffix == ".dae":
        return _load_dae_vertices(mesh_path)
    raise ValueError(f"Unsupported mesh type for proxy generation: {mesh_path}")


def _load_binary_stl_vertices(mesh_path: Path) -> np.ndarray:
    import struct

    data = mesh_path.read_bytes()
    triangle_count = struct.unpack("<I", data[80:84])[0]
    vertices = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack("<12fH", data[offset : offset + 50])
        vertices.extend([values[3:6], values[6:9], values[9:12]])
        offset += 50
    return np.asarray(vertices, dtype=float)


def _load_dae_vertices(mesh_path: Path) -> np.ndarray:
    root = ET.parse(mesh_path).getroot()
    components = []
    for geometry in root.iter():
        if not geometry.tag.endswith("geometry"):
            continue
        geometry_points = []
        for float_array in geometry.iter():
            if not float_array.tag.endswith("float_array") or not float_array.text:
                continue
            array_id = float_array.attrib.get("id", "").lower()
            if "position" not in array_id:
                continue
            values = np.fromstring(float_array.text, sep=" ", dtype=float)
            if len(values) >= 3 and len(values) % 3 == 0:
                points = values.reshape(-1, 3)
                if np.max(np.ptp(points, axis=0)) > 1e-6:
                    geometry_points.append(points)
        if geometry_points:
            points = np.vstack(geometry_points)
            size = np.ptp(points, axis=0)
            components.append((points, len(points), float(np.prod(np.maximum(size, 1e-6)))))
    if not components:
        raise ValueError(f"No vertices found in {mesh_path}")
    max_count = max(count for _, count, _ in components)
    max_volume = max(volume for _, _, volume in components)
    kept = [
        points
        for points, count, volume in components
        if count >= 0.02 * max_count or volume >= 0.02 * max_volume
    ]
    return np.vstack(kept)


def _seed_joint_configuration(elbow_up: bool) -> np.ndarray:
    if elbow_up:
        return np.array([0.0, 1.15, -1.35, 0.0, 0.45, 0.0], dtype=float)
    return np.array([0.0, 0.85, -1.05, 0.0, 0.25, 0.0], dtype=float)


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=float)


def _transform_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return out
