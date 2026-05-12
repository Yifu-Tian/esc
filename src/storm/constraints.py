from __future__ import annotations

import numpy as np

from storm.core import ConstraintReport


def goal_reached_report(endpoint: np.ndarray, goal: np.ndarray, tolerance: float, name: str = "goal_error") -> ConstraintReport:
    error = float(np.linalg.norm(np.asarray(endpoint, dtype=float) - np.asarray(goal, dtype=float)))
    return ConstraintReport(name=name, value=error, threshold=float(tolerance), feasible=error <= tolerance)


def smoothness_cost(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    accel = points[2:] - 2.0 * points[1:-1] + points[:-2]
    return float(np.mean(np.sum(accel * accel, axis=1)))


def circle_collision_report(
    points_xy: np.ndarray,
    centers_xy: np.ndarray,
    radii: np.ndarray,
    robot_radius: float = 0.0,
    name: str = "circle_collision",
) -> ConstraintReport:
    points = np.asarray(points_xy, dtype=float)
    centers = np.asarray(centers_xy, dtype=float)
    radii = np.asarray(radii, dtype=float) + float(robot_radius)
    if len(centers) == 0:
        return ConstraintReport(name=name, value=float("inf"), feasible=True, threshold=0.0)
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
    min_clearance = float(np.min(distances))
    return ConstraintReport(name=name, value=min_clearance, feasible=min_clearance >= 0.0, threshold=0.0)
