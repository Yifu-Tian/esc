from __future__ import annotations

import numpy as np

from mm_flow.three_d.collision import (
    CuboidObstacle,
    CylinderObstacle,
    Obstacle3D,
    SphereObstacle,
    state_clearance,
)
from mm_flow.three_d.heuristic_planner import plan_with_geometric_seeds_3d
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.rrt_connect import RRTConnectConfig, plan_rrt_connect_to_goal


def build_reach_sequence_scene_3d(
    seed: int = 7,
    random_obstacle_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[Obstacle3D]]:
    start = np.array([-2.25, -1.20, 0.12, 0.2, 0.55, -1.45], dtype=float)
    goals_xyz = np.array(
        [
            [-1.05, 0.92, 0.85],
            [0.75, -0.82, 1.10],
            [2.05, 0.88, 0.72],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    robot = SimpleMobileManipulator3D()
    obstacles = _sample_random_obstacles(rng, random_obstacle_count, start, goals_xyz, robot)
    return start, goals_xyz, obstacles


def plan_reach_sequence_3d(
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goals_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    segment_horizon: int,
    planner: str = "heuristic",
    rrt_config: RRTConnectConfig | None = None,
) -> tuple[np.ndarray, list[int], list[float], list[float], bool]:
    current = start.copy()
    pieces = []
    segment_ends: list[int] = []
    terminal_errors: list[float] = []
    clearances: list[float] = []
    success = True

    for i, goal_xyz in enumerate(goals_xyz):
        if planner == "heuristic":
            result = plan_with_geometric_seeds_3d(
                robot=robot,
                start=current,
                goal_xyz=goal_xyz,
                obstacles=obstacles,
                horizon=segment_horizon,
            )
        elif planner == "rrt_connect":
            config = rrt_config or RRTConnectConfig()
            config = RRTConnectConfig(
                max_iterations=config.max_iterations,
                step_size=config.step_size,
                edge_resolution=config.edge_resolution,
                goal_tolerance=config.goal_tolerance,
                goal_sample_count=config.goal_sample_count,
                shortcut_attempts=config.shortcut_attempts,
                rng_seed=config.rng_seed + i * 997,
                clearance_margin=config.clearance_margin,
                bounds_xy=config.bounds_xy,
            )
            result = plan_rrt_connect_to_goal(
                robot=robot,
                start=current,
                goal_xyz=goal_xyz,
                obstacles=obstacles,
                config=config,
            )
        else:
            raise ValueError(f"Unsupported planner: {planner}")
        segment = result.trajectory if i == 0 else result.trajectory[1:]
        pieces.append(segment)
        current = result.trajectory[-1].copy()
        terminal_errors.append(result.terminal_error)
        clearances.append(result.min_clearance)
        segment_ends.append(sum(len(piece) for piece in pieces) - 1)
        success = success and result.success
        if i < len(goals_xyz) - 1:
            dwell = np.repeat(current[None, :], 4, axis=0)
            pieces.append(dwell)

    trajectory = np.vstack(pieces)
    trajectory[:, 2] = np.unwrap(trajectory[:, 2])
    trajectory[:, 3] = np.unwrap(trajectory[:, 3])
    return trajectory, segment_ends, terminal_errors, clearances, bool(success)


def densify_trajectory_for_visualization(
    trajectory: np.ndarray,
    segment_ends: list[int],
    max_state_step: float = 0.12,
) -> tuple[np.ndarray, list[int]]:
    dense = [trajectory[0]]
    index_map = {0: 0}
    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]
        delta = end - start
        steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_state_step)))
        for k in range(1, steps + 1):
            alpha = k / steps
            dense.append((1.0 - alpha) * start + alpha * end)
        index_map[i + 1] = len(dense) - 1
    dense_segment_ends = [index_map[end] for end in segment_ends]
    return np.vstack(dense), dense_segment_ends


def _sample_random_obstacles(
    rng: np.random.Generator,
    count: int,
    start: np.ndarray,
    goals_xyz: np.ndarray,
    robot: SimpleMobileManipulator3D,
) -> list[Obstacle3D]:
    obstacles: list[Obstacle3D] = []
    protected_xy = [start[:2], *[goal[:2] for goal in goals_xyz]]

    attempts = 0
    while len(obstacles) < count and attempts < count * 80:
        attempts += 1
        kind = rng.choice(["sphere", "cube", "cuboid", "cylinder"], p=[0.28, 0.22, 0.30, 0.20])
        xy = rng.uniform([-2.05, -1.35], [2.05, 1.35])
        if min(np.linalg.norm(xy - p) for p in protected_xy) < 0.55:
            continue

        grounded = bool(rng.random() < 0.58)
        if kind == "sphere":
            radius = float(rng.uniform(0.16, 0.34))
            z = radius if grounded else float(rng.uniform(0.55, 1.45))
            obstacle: Obstacle3D = SphereObstacle(center=(float(xy[0]), float(xy[1]), z), radius=radius)
        elif kind == "cube":
            side = float(rng.uniform(0.28, 0.58))
            z = 0.5 * side if grounded else float(rng.uniform(0.55, 1.25))
            obstacle = CuboidObstacle(center=(float(xy[0]), float(xy[1]), z), size=(side, side, side))
        elif kind == "cuboid":
            size = rng.uniform([0.24, 0.24, 0.35], [0.70, 0.90, 1.25])
            z = 0.5 * float(size[2]) if grounded else float(rng.uniform(0.55, 1.35))
            obstacle = CuboidObstacle(
                center=(float(xy[0]), float(xy[1]), z),
                size=(float(size[0]), float(size[1]), float(size[2])),
            )
        else:
            radius = float(rng.uniform(0.15, 0.32))
            height = float(rng.uniform(0.45, 1.35))
            z = 0.5 * height if grounded else float(rng.uniform(0.65, 1.35))
            obstacle = CylinderObstacle(center=(float(xy[0]), float(xy[1]), z), radius=radius, height=height)

        if _too_close_to_existing(obstacle, obstacles):
            continue
        if state_clearance(robot, start, [*obstacles, obstacle]) < 0.12:
            continue
        obstacles.append(obstacle)

    return obstacles


def _too_close_to_existing(candidate: Obstacle3D, existing: list[Obstacle3D], min_distance: float = 0.18) -> bool:
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
        radius = float(np.hypot(obstacle.radius, 0.5 * obstacle.height))
        return np.array(obstacle.center, dtype=float), radius
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")
