from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from mm_flow.three_d.kinematics import SimpleMobileManipulator3D


@dataclass(frozen=True)
class SphereObstacle:
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class CuboidObstacle:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class CylinderObstacle:
    center: tuple[float, float, float]
    radius: float
    height: float


Obstacle3D = Union[SphereObstacle, CuboidObstacle, CylinderObstacle]


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = np.clip(float(np.dot(point - a, ab) / denom), 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def point_box_distance(point: np.ndarray, box: CuboidObstacle) -> float:
    center = np.array(box.center, dtype=float)
    half = 0.5 * np.array(box.size, dtype=float)
    delta = np.abs(point - center) - half
    outside = np.maximum(delta, 0.0)
    outside_dist = np.linalg.norm(outside)
    inside_dist = min(float(np.max(delta)), 0.0)
    return float(outside_dist + inside_dist)


def segment_box_distance(a: np.ndarray, b: np.ndarray, box: CuboidObstacle, samples: int = 13) -> float:
    alphas = np.linspace(0.0, 1.0, samples)
    return min(point_box_distance((1.0 - t) * a + t * b, box) for t in alphas)


def point_cylinder_distance(point: np.ndarray, cylinder: CylinderObstacle) -> float:
    center = np.array(cylinder.center, dtype=float)
    radial = np.linalg.norm(point[:2] - center[:2]) - cylinder.radius
    vertical = abs(float(point[2] - center[2])) - 0.5 * cylinder.height
    outside = np.array([max(radial, 0.0), max(vertical, 0.0)])
    outside_dist = np.linalg.norm(outside)
    inside_dist = min(max(radial, vertical), 0.0)
    return float(outside_dist + inside_dist)


def segment_cylinder_distance(a: np.ndarray, b: np.ndarray, cylinder: CylinderObstacle, samples: int = 13) -> float:
    alphas = np.linspace(0.0, 1.0, samples)
    return min(point_cylinder_distance((1.0 - t) * a + t * b, cylinder) for t in alphas)


def state_clearance(
    robot: SimpleMobileManipulator3D,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
) -> float:
    if hasattr(robot, "export_collision_proxies") and hasattr(robot, "link_transforms"):
        return _proxy_state_clearance(robot, state, obstacles)
    return _segment_state_clearance(robot, state, obstacles)


def is_state_collision_free(
    robot: SimpleMobileManipulator3D,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
    clearance_margin: float = 0.0,
) -> bool:
    if hasattr(robot, "export_collision_proxies") and hasattr(robot, "link_transforms"):
        return _proxy_state_collision_free(robot, state, obstacles, clearance_margin)
    return state_clearance(robot, state, obstacles) >= clearance_margin


def _segment_state_clearance(
    robot: SimpleMobileManipulator3D,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
) -> float:
    fk = robot.forward_kinematics(state)
    base_center = fk["base_center"]
    base_sphere_radius = max(robot.base_radius, 0.5 * robot.base_height)
    min_clearance = np.inf

    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            center = np.array(obs.center, dtype=float)
            base_clearance = np.linalg.norm(base_center - center) - base_sphere_radius - obs.radius
            min_clearance = min(min_clearance, float(base_clearance))
            for segment in fk["segments"]:
                a = segment[:3]
                b = segment[3:]
                link_clearance = point_segment_distance(center, a, b) - robot.link_radius - obs.radius
                min_clearance = min(min_clearance, float(link_clearance))
        elif isinstance(obs, CuboidObstacle):
            base_clearance = point_box_distance(base_center, obs) - base_sphere_radius
            min_clearance = min(min_clearance, float(base_clearance))
            for segment in fk["segments"]:
                a = segment[:3]
                b = segment[3:]
                link_clearance = segment_box_distance(a, b, obs) - robot.link_radius
                min_clearance = min(min_clearance, float(link_clearance))
        elif isinstance(obs, CylinderObstacle):
            base_clearance = point_cylinder_distance(base_center, obs) - base_sphere_radius
            min_clearance = min(min_clearance, float(base_clearance))
            for segment in fk["segments"]:
                a = segment[:3]
                b = segment[3:]
                link_clearance = segment_cylinder_distance(a, b, obs) - robot.link_radius
                min_clearance = min(min_clearance, float(link_clearance))
        else:
            raise TypeError(f"Unsupported obstacle type: {type(obs)!r}")

    ground_clearance = min(fk["mount"][2], fk["elbow"][2], fk["ee"][2]) - robot.link_radius
    min_clearance = min(min_clearance, float(ground_clearance))
    return float(min_clearance)


def _proxy_state_collision_free(
    robot: Any,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
    clearance_margin: float,
) -> bool:
    link_transforms = robot.link_transforms(state)
    for proxy in robot.export_collision_proxies():
        if proxy["type"] != "box":
            raise TypeError(f"Unsupported collision proxy type: {proxy['type']!r}")
        link_matrix = link_transforms.get(proxy["link"])
        if link_matrix is None:
            continue
        box = _proxy_to_obb(proxy, link_matrix)
        if proxy["link"] != "mobile_base" and _box_ground_clearance(box) < clearance_margin:
            return False
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                if _obb_sphere_distance(box, np.asarray(obs.center, dtype=float), obs.radius) < clearance_margin:
                    return False
            elif isinstance(obs, CuboidObstacle):
                inflated = _inflate_cuboid(obs, clearance_margin)
                if _obb_obb_overlap(box, _cuboid_to_obb(inflated))[0]:
                    return False
            elif isinstance(obs, CylinderObstacle):
                inflated = CuboidObstacle(
                    center=obs.center,
                    size=(
                        2.0 * (obs.radius + clearance_margin),
                        2.0 * (obs.radius + clearance_margin),
                        obs.height + 2.0 * clearance_margin,
                    ),
                )
                if _obb_obb_overlap(box, _cuboid_to_obb(inflated))[0]:
                    return False
            else:
                raise TypeError(f"Unsupported obstacle type: {type(obs)!r}")
    return True


def _proxy_state_clearance(
    robot: Any,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
) -> float:
    link_transforms = robot.link_transforms(state)
    min_clearance = np.inf
    for proxy in robot.export_collision_proxies():
        if proxy["type"] != "box":
            raise TypeError(f"Unsupported collision proxy type: {proxy['type']!r}")
        link_matrix = link_transforms.get(proxy["link"])
        if link_matrix is None:
            continue
        box = _proxy_to_obb(proxy, link_matrix)
        if proxy["link"] != "mobile_base":
            min_clearance = min(min_clearance, float(_box_ground_clearance(box)))
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                clearance = _obb_sphere_distance(box, np.asarray(obs.center, dtype=float), obs.radius)
            elif isinstance(obs, CuboidObstacle):
                clearance = _obb_obb_distance(box, _cuboid_to_obb(obs))
            elif isinstance(obs, CylinderObstacle):
                # Conservative proxy: a vertical cylinder is contained in this AABB.
                cylinder_box = CuboidObstacle(
                    center=obs.center,
                    size=(2.0 * obs.radius, 2.0 * obs.radius, obs.height),
                )
                clearance = _obb_obb_distance(box, _cuboid_to_obb(cylinder_box))
            else:
                raise TypeError(f"Unsupported obstacle type: {type(obs)!r}")
            min_clearance = min(min_clearance, float(clearance))
    return float(min_clearance)


def _proxy_to_obb(proxy: dict[str, Any], link_matrix: np.ndarray) -> "_OBB":
    proxy_matrix = np.asarray(proxy["originMatrix"], dtype=float)
    world_matrix = link_matrix @ proxy_matrix
    return _OBB(
        center=world_matrix[:3, 3].copy(),
        axes=world_matrix[:3, :3].copy(),
        half_size=0.5 * np.asarray(proxy["size"], dtype=float),
    )


def trajectory_min_clearance(
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    obstacles: list[Obstacle3D],
) -> float:
    return min(state_clearance(robot, state, obstacles) for state in trajectory)


@dataclass(frozen=True)
class _OBB:
    center: np.ndarray
    axes: np.ndarray
    half_size: np.ndarray


_BOX_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=float,
)

_BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _cuboid_to_obb(box: CuboidObstacle) -> _OBB:
    return _OBB(
        center=np.asarray(box.center, dtype=float),
        axes=np.eye(3),
        half_size=0.5 * np.asarray(box.size, dtype=float),
    )


def _inflate_cuboid(box: CuboidObstacle, margin: float) -> CuboidObstacle:
    size = tuple(float(v + 2.0 * margin) for v in box.size)
    return CuboidObstacle(center=box.center, size=size)


def _box_corners(box: _OBB) -> np.ndarray:
    return box.center + (_BOX_SIGNS * box.half_size) @ box.axes.T


def _box_edges(box: _OBB) -> list[tuple[np.ndarray, np.ndarray]]:
    corners = _box_corners(box)
    return [(corners[i], corners[j]) for i, j in _BOX_EDGES]


def _box_ground_clearance(box: _OBB) -> float:
    return float(np.min(_box_corners(box)[:, 2]))


def _signed_point_obb_distance(point: np.ndarray, box: _OBB) -> float:
    local = box.axes.T @ (point - box.center)
    delta = np.abs(local) - box.half_size
    outside = np.maximum(delta, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = min(float(np.max(delta)), 0.0)
    return float(outside_dist + inside_dist)


def _obb_sphere_distance(box: _OBB, center: np.ndarray, radius: float) -> float:
    return float(_signed_point_obb_distance(center, box) - radius)


def _obb_obb_distance(a: _OBB, b: _OBB) -> float:
    collision, penetration = _obb_obb_overlap(a, b)
    if collision:
        return -float(penetration)

    corners_a = _box_corners(a)
    corners_b = _box_corners(b)
    distances = [
        *(_signed_point_obb_distance(p, b) for p in corners_a),
        *(_signed_point_obb_distance(p, a) for p in corners_b),
    ]
    for a0, a1 in _box_edges(a):
        for b0, b1 in _box_edges(b):
            distances.append(_segment_segment_distance(a0, a1, b0, b1))
    return float(min(distances))


def _obb_obb_overlap(a: _OBB, b: _OBB) -> tuple[bool, float]:
    axes = [a.axes[:, i] for i in range(3)] + [b.axes[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(a.axes[:, i], b.axes[:, j])
            norm = float(np.linalg.norm(cross))
            if norm > 1e-9:
                axes.append(cross / norm)

    min_overlap = np.inf
    for axis in axes:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        a_radius = float(np.sum(a.half_size * np.abs(a.axes.T @ axis)))
        b_radius = float(np.sum(b.half_size * np.abs(b.axes.T @ axis)))
        center_distance = abs(float(np.dot(b.center - a.center, axis)))
        overlap = a_radius + b_radius - center_distance
        if overlap < 0.0:
            return False, 0.0
        min_overlap = min(min_overlap, overlap)
    return True, float(min_overlap)


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-12:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t_nom = b * s + f
            if t_nom < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t_nom > e:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
            else:
                t = t_nom / e

    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    return float(np.linalg.norm(closest1 - closest2))
