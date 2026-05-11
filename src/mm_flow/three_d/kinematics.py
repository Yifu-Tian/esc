from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimpleMobileManipulator3D:
    """Ground mobile base plus a 3-DoF spatial arm.

    State is [x, y, yaw, q1, q2, q3].
    The base moves on the ground plane. The arm has one yaw joint and two
    pitch joints, which is enough for a first 3D reaching scaffold.
    """

    link_lengths: tuple[float, float] = (0.9, 0.65)
    mount_height: float = 0.45
    base_radius: float = 0.32
    base_height: float = 0.32
    link_radius: float = 0.045
    q_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-2.8, 2.8),
        (-1.35, 1.35),
        (-2.35, 2.35),
    )

    @property
    def state_dim(self) -> int:
        return 6

    def forward_kinematics(self, state: np.ndarray) -> dict[str, np.ndarray]:
        x, y, yaw, q1, q2, q3 = state
        l1, l2 = self.link_lengths
        base = np.array([x, y, 0.0], dtype=float)
        mount = np.array([x, y, self.mount_height], dtype=float)
        azimuth = yaw + q1

        dir1 = np.array([
            np.cos(azimuth) * np.cos(q2),
            np.sin(azimuth) * np.cos(q2),
            np.sin(q2),
        ])
        elbow = mount + l1 * dir1
        pitch2 = q2 + q3
        dir2 = np.array([
            np.cos(azimuth) * np.cos(pitch2),
            np.sin(azimuth) * np.cos(pitch2),
            np.sin(pitch2),
        ])
        ee = elbow + l2 * dir2

        return {
            "base": base,
            "base_center": np.array([x, y, 0.5 * self.base_height], dtype=float),
            "mount": mount,
            "elbow": elbow,
            "ee": ee,
            "segments": np.array([
                [mount[0], mount[1], mount[2], elbow[0], elbow[1], elbow[2]],
                [elbow[0], elbow[1], elbow[2], ee[0], ee[1], ee[2]],
            ]),
        }

    def end_effector(self, state: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(state)["ee"]

    def inverse_kinematics_for_goal(
        self,
        base_xy: np.ndarray,
        yaw: float,
        goal_xyz: np.ndarray,
        elbow_up: bool = False,
    ) -> np.ndarray:
        l1, l2 = self.link_lengths
        mount = np.array([base_xy[0], base_xy[1], self.mount_height], dtype=float)
        rel_world = goal_xyz - mount
        q1 = np.arctan2(rel_world[1], rel_world[0]) - yaw
        horizontal = np.linalg.norm(rel_world[:2])
        z = rel_world[2]
        r = np.hypot(horizontal, z)
        r = np.clip(r, 1e-6, l1 + l2 - 1e-6)
        c3 = np.clip((r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        q3 = np.arccos(c3)
        if not elbow_up:
            q3 = -q3
        q2 = np.arctan2(z, horizontal) - np.arctan2(l2 * np.sin(q3), l1 + l2 * np.cos(q3))
        return self.clip_joints(np.array([wrap_to_pi(q1), q2, q3], dtype=float))

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        lo = np.array([lim[0] for lim in self.q_limits])
        hi = np.array([lim[1] for lim in self.q_limits])
        return np.clip(q, lo, hi)


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def smooth_1d(values: np.ndarray, passes: int = 1) -> np.ndarray:
    out = values.copy()
    for _ in range(passes):
        prev = out.copy()
        out[1:-1] = 0.25 * prev[:-2] + 0.5 * prev[1:-1] + 0.25 * prev[2:]
    return out

