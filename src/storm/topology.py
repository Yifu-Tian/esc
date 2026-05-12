from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from storm.core import ConstraintReport


@dataclass(frozen=True)
class CircularObstacle2D:
    center: np.ndarray
    radius: float
    name: str = "obstacle"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,):
            raise ValueError(f"center must have shape (2,), got {center.shape}")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        object.__setattr__(self, "center", center)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def concatenate_history_candidate(history_xy: np.ndarray | None, candidate_xy: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate_xy, dtype=float)
    if history_xy is None:
        return candidate
    history = np.asarray(history_xy, dtype=float)
    if len(history) == 0:
        return candidate
    if len(candidate) == 0:
        return history
    if np.linalg.norm(history[-1] - candidate[0]) < 1e-9:
        return np.vstack([history, candidate[1:]])
    return np.vstack([history, candidate])


def fractional_winding_number(points_xy: np.ndarray, center_xy: np.ndarray) -> float:
    """Approximate the open-curve fractional winding number around a 2D point."""

    points = np.asarray(points_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {points.shape}")
    if center.shape != (2,):
        raise ValueError(f"center_xy must have shape (2,), got {center.shape}")
    if len(points) < 2:
        return 0.0
    vectors = points - center[None, :]
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-9):
        return float("inf")
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float((angles[-1] - angles[0]) / (2.0 * np.pi))


def topology_reports(
    history_xy: np.ndarray | None,
    candidate_xy: np.ndarray,
    obstacles: list[CircularObstacle2D],
    winding_threshold: float = 0.95,
) -> tuple[ConstraintReport, ...]:
    global_path = concatenate_history_candidate(history_xy, candidate_xy)
    reports: list[ConstraintReport] = []
    max_abs_winding = 0.0
    worst_obstacle = ""
    for obstacle in obstacles:
        winding = fractional_winding_number(global_path, obstacle.center)
        abs_winding = abs(winding)
        if abs_winding > max_abs_winding:
            max_abs_winding = abs_winding
            worst_obstacle = obstacle.name
        reports.append(
            ConstraintReport(
                name=f"winding/{obstacle.name}",
                value=float(winding),
                feasible=abs_winding < winding_threshold,
                threshold=float(winding_threshold),
                metadata={"radius": obstacle.radius},
            )
        )
    reports.append(
        ConstraintReport(
            name="max_abs_winding",
            value=float(max_abs_winding),
            feasible=max_abs_winding < winding_threshold,
            threshold=float(winding_threshold),
            metadata={"worst_obstacle": worst_obstacle},
        )
    )
    return tuple(reports)
