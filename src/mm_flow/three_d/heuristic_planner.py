from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_flow.three_d.collision import Obstacle3D, trajectory_min_clearance
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D, smooth_1d


@dataclass(frozen=True)
class HeuristicPlanResult3D:
    trajectory: np.ndarray
    success: bool
    terminal_error: float
    min_clearance: float
    candidate_index: int


def plan_with_geometric_seeds_3d(
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goal_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    horizon: int = 44,
) -> HeuristicPlanResult3D:
    start_xy = start[:2]
    goal_xy = goal_xyz[:2]
    direction = goal_xy - start_xy
    if np.linalg.norm(direction) < 1e-6:
        direction_unit = np.array([np.cos(start[2]), np.sin(start[2])])
    else:
        direction_unit = direction / np.linalg.norm(direction)

    approach_angle = float(np.arctan2(direction_unit[1], direction_unit[0]))
    standoff = 0.95
    final_base_candidates = [
        goal_xy - standoff * np.array([np.cos(approach_angle + delta), np.sin(approach_angle + delta)])
        for delta in [-1.45, -0.95, -0.45, 0.0, 0.45, 0.95, 1.45]
    ]

    base_candidates: list[np.ndarray] = []
    for final_base in final_base_candidates:
        local_direction = final_base - start_xy
        if np.linalg.norm(local_direction) < 1e-6:
            local_unit = direction_unit
        else:
            local_unit = local_direction / np.linalg.norm(local_direction)
        normal = np.array([-local_unit[1], local_unit[0]])
        for offset in [-1.7, -1.15, -0.65, 0.0, 0.65, 1.15, 1.7]:
            control = 0.5 * (start_xy + final_base) + offset * normal
            base_candidates.append(_quadratic_bezier(start_xy, control, final_base, horizon))
        for offset in [-1.55, 1.55]:
            p1 = start_xy + 0.35 * (final_base - start_xy) + offset * normal
            p2 = start_xy + 0.72 * (final_base - start_xy) + offset * normal
            base_candidates.append(_polyline([start_xy, p1, p2, final_base], horizon))

    best_traj = None
    best_score = -np.inf
    best_idx = -1
    best_clearance = -np.inf
    best_terminal = np.inf

    for idx, base_path in enumerate(base_candidates):
        traj = _build_whole_body_trajectory(robot, start, goal_xyz, base_path)
        clearance = trajectory_min_clearance(robot, traj, obstacles)
        terminal = float(np.linalg.norm(robot.end_effector(traj[-1]) - goal_xyz))
        smoothness = float(np.sum(np.diff(traj, axis=0) ** 2))
        score = 8.0 * clearance - 0.8 * terminal - 0.015 * smoothness
        if score > best_score:
            best_score = score
            best_traj = traj
            best_idx = idx
            best_clearance = clearance
            best_terminal = terminal

    assert best_traj is not None
    success = bool(best_terminal < 0.09 and best_clearance > -1e-3)
    return HeuristicPlanResult3D(best_traj, success, best_terminal, best_clearance, best_idx)


def _build_whole_body_trajectory(
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goal_xyz: np.ndarray,
    base_path: np.ndarray,
) -> np.ndarray:
    horizon = len(base_path)
    traj = np.zeros((horizon, robot.state_dim), dtype=float)
    traj[:, :2] = base_path

    delta = np.gradient(base_path, axis=0)
    yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    yaw[0] = start[2]
    traj[:, 2] = smooth_1d(yaw, passes=2)

    compact_q = np.array([0.0, 0.65, -1.85], dtype=float)
    final_q = robot.inverse_kinematics_for_goal(base_path[-1], traj[-1, 2], goal_xyz, elbow_up=False)
    progress = np.linspace(0.0, 1.0, horizon)
    for i, p in enumerate(progress):
        if p < 0.66:
            a = p / 0.66
            q = (1.0 - a) * start[3:6] + a * compact_q
        else:
            a = (p - 0.66) / 0.34
            a = 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0.0, 1.0))
            q = (1.0 - a) * compact_q + a * final_q
        traj[i, 3:6] = robot.clip_joints(q)
    traj[0] = start
    return traj



def _quadratic_bezier(start: np.ndarray, control: np.ndarray, end: np.ndarray, num: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num)[:, None]
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t ** 2 * end


def _polyline(points: list[np.ndarray], num: int) -> np.ndarray:
    pts = np.array(points, dtype=float)
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], num, axis=0)
    samples = np.linspace(0.0, total, num)
    out = np.zeros((num, 2), dtype=float)
    for i, s in enumerate(samples):
        seg = min(np.searchsorted(cumulative, s, side="right") - 1, len(lengths) - 1)
        local = (s - cumulative[seg]) / max(lengths[seg], 1e-9)
        out[i] = (1.0 - local) * pts[seg] + local * pts[seg + 1]
    return out
