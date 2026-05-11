from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, Obstacle3D, SphereObstacle, state_clearance
from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D


@dataclass(frozen=True)
class ReachSequenceProblem:
    name: str
    robot: MomaPiperMobileManipulator3D
    start: np.ndarray
    goals_xyz: np.ndarray
    obstacles: list[Obstacle3D]
    seed: int
    bounds_xy: tuple[tuple[float, float], tuple[float, float]]
    task_type: str = "reach_sequence"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def goal_count(self) -> int:
        return int(len(self.goals_xyz))

    @property
    def obstacle_count(self) -> int:
        return int(len(self.obstacles))


def build_moma_reach_sequence_problem(
    robot: MomaPiperMobileManipulator3D | None = None,
    seed: int = 3,
    obstacle_count: int = 24,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]] = ((-2.4, 2.4), (-1.7, 1.7)),
) -> ReachSequenceProblem:
    robot = robot or MomaPiperMobileManipulator3D()
    start_q = np.array([0.0, 0.8, -1.1, 0.0, 0.25, 0.0], dtype=float)
    start = np.array([-1.45, -0.75, 0.15, *robot.clip_joints(start_q)], dtype=float)
    goals_xyz = np.array(
        [
            [-0.42, 0.70, 0.74],
            [0.82, 0.58, 0.72],
            [1.28, -0.42, 0.82],
        ],
        dtype=float,
    )
    obstacles = _sample_spread_obstacles(np.random.default_rng(seed), obstacle_count, robot, start, goals_xyz)
    return ReachSequenceProblem(
        name="moma_reach_sequence",
        robot=robot,
        start=start,
        goals_xyz=goals_xyz,
        obstacles=obstacles,
        seed=seed,
        bounds_xy=bounds_xy,
        metadata={"robot": "moma_piper_9dof", "obstacle_count": obstacle_count},
    )


def build_dynamic_obstacle_problem(*args, **kwargs) -> ReachSequenceProblem:
    raise NotImplementedError("DynamicObstacleProblem is planned but not implemented yet.")


def build_narrow_passage_problem(*args, **kwargs) -> ReachSequenceProblem:
    raise NotImplementedError("NarrowPassageProblem is planned but not implemented yet.")


def build_replanning_problem(*args, **kwargs) -> ReachSequenceProblem:
    raise NotImplementedError("ReplanningProblem is planned but not implemented yet.")


def _sample_spread_obstacles(
    rng: np.random.Generator,
    count: int,
    robot: MomaPiperMobileManipulator3D,
    start: np.ndarray,
    goals_xyz: np.ndarray,
) -> list[Obstacle3D]:
    obstacles: list[Obstacle3D] = _sample_corridor_obstacles(rng, count, robot, start, goals_xyz)
    protected_xy = [start[:2], *[goal[:2] for goal in goals_xyz], np.array([-0.25, -0.05]), np.array([0.35, 0.35])]
    attempts = 0
    while len(obstacles) < count and attempts < count * 150:
        attempts += 1
        kind = rng.choice(["sphere", "cube", "cuboid", "cylinder"], p=[0.30, 0.18, 0.30, 0.22])
        xy = rng.uniform([-2.05, -1.45], [2.05, 1.45])

        if min(np.linalg.norm(xy - p) for p in protected_xy) < 0.62:
            continue

        grounded = bool(rng.random() < 0.62)
        if kind == "sphere":
            radius = float(rng.uniform(0.12, 0.24))
            z = radius if grounded else float(rng.uniform(0.55, 1.10))
            obstacle: Obstacle3D = SphereObstacle(center=(float(xy[0]), float(xy[1]), z), radius=radius)
        elif kind == "cube":
            side = float(rng.uniform(0.22, 0.42))
            z = 0.5 * side if grounded else float(rng.uniform(0.55, 1.05))
            obstacle = CuboidObstacle(center=(float(xy[0]), float(xy[1]), z), size=(side, side, side))
        elif kind == "cuboid":
            size = rng.uniform([0.22, 0.22, 0.28], [0.55, 0.70, 0.95])
            z = 0.5 * float(size[2]) if grounded else float(rng.uniform(0.58, 1.12))
            obstacle = CuboidObstacle(
                center=(float(xy[0]), float(xy[1]), z),
                size=(float(size[0]), float(size[1]), float(size[2])),
            )
        else:
            radius = float(rng.uniform(0.12, 0.24))
            height = float(rng.uniform(0.35, 1.00))
            z = 0.5 * height if grounded else float(rng.uniform(0.62, 1.15))
            obstacle = CylinderObstacle(center=(float(xy[0]), float(xy[1]), z), radius=radius, height=height)

        if _too_close_to_existing(obstacle, obstacles, min_distance=0.24):
            continue
        if state_clearance(robot, start, [*obstacles, obstacle]) < 0.12:
            continue
        obstacles.append(obstacle)
    return obstacles


def _sample_corridor_obstacles(
    rng: np.random.Generator,
    count: int,
    robot: MomaPiperMobileManipulator3D,
    start: np.ndarray,
    goals_xyz: np.ndarray,
) -> list[Obstacle3D]:
    obstacles: list[Obstacle3D] = []
    segment_points = [start[:2], *goals_xyz[:, :2]]
    max_corridor_obstacles = min(count, 2 * (len(segment_points) - 1))
    for i, (a, b) in enumerate(zip(segment_points[:-1], segment_points[1:])):
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        normal = np.array([-direction[1], direction[0]], dtype=float) / length
        for alpha in (0.38, 0.66):
            if len(obstacles) >= max_corridor_obstacles:
                return obstacles
            center_xy = (1.0 - alpha) * a + alpha * b
            center_xy = center_xy + normal * float(rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.18))
            obstacle = _make_corridor_obstacle(rng, center_xy, i)
            if state_clearance(robot, start, [*obstacles, obstacle]) < 0.12:
                continue
            if _too_close_to_existing(obstacle, obstacles, min_distance=0.10):
                continue
            obstacles.append(obstacle)
    return obstacles


def _make_corridor_obstacle(rng: np.random.Generator, xy: np.ndarray, index: int) -> Obstacle3D:
    if index % 3 == 0:
        radius = float(rng.uniform(0.18, 0.28))
        height = float(rng.uniform(0.70, 1.25))
        return CylinderObstacle(center=(float(xy[0]), float(xy[1]), 0.5 * height), radius=radius, height=height)
    if index % 3 == 1:
        size = rng.uniform([0.30, 0.24, 0.45], [0.58, 0.46, 1.05])
        return CuboidObstacle(
            center=(float(xy[0]), float(xy[1]), 0.5 * float(size[2])),
            size=(float(size[0]), float(size[1]), float(size[2])),
        )
    radius = float(rng.uniform(0.18, 0.30))
    return SphereObstacle(center=(float(xy[0]), float(xy[1]), radius), radius=radius)


def _too_close_to_existing(candidate: Obstacle3D, existing: list[Obstacle3D], min_distance: float) -> bool:
    c_center, c_radius = _bounding_sphere(candidate)
    for obstacle in existing:
        center, radius = _bounding_sphere(obstacle)
        if np.linalg.norm(c_center - center) < c_radius + radius + min_distance:
            return True
    return False


def _bounding_sphere(obstacle: Obstacle3D) -> tuple[np.ndarray, float]:
    if isinstance(obstacle, SphereObstacle):
        return np.array(obstacle.center, dtype=float), obstacle.radius
    if isinstance(obstacle, CuboidObstacle):
        return np.array(obstacle.center, dtype=float), 0.5 * float(np.linalg.norm(obstacle.size))
    if isinstance(obstacle, CylinderObstacle):
        return np.array(obstacle.center, dtype=float), float(np.hypot(obstacle.radius, 0.5 * obstacle.height))
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")
