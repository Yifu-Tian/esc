#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, Obstacle3D, SphereObstacle, state_clearance
from mm_flow.three_d.html_viewer import export_reach_sequence_html
from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D
from mm_flow.three_d.planners import available_planners, get_planner, plan_reach_sequence
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.tasks import densify_trajectory_for_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a whole-body planner on the URDF-derived MoMa/Piper scaffold.")
    parser.add_argument("--planner", choices=available_planners(), default="rrt_connect")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=9000)
    parser.add_argument("--goal-samples", type=int, default=180)
    parser.add_argument("--step-size", type=float, default=0.18)
    parser.add_argument("--edge-resolution", type=float, default=0.06)
    parser.add_argument("--clearance-margin", type=float, default=0.01)
    parser.add_argument("--obstacle-count", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moma_rrt_connect"))
    args = parser.parse_args()

    robot = MomaPiperMobileManipulator3D()
    start, goals_xyz, obstacles = build_moma_demo_scene(seed=args.seed, obstacle_count=args.obstacle_count)
    config = RRTConnectConfig(
        max_iterations=args.max_iterations,
        step_size=args.step_size,
        edge_resolution=args.edge_resolution,
        goal_sample_count=args.goal_samples,
        shortcut_attempts=80,
        rng_seed=args.seed,
        clearance_margin=args.clearance_margin,
        bounds_xy=((-2.4, 2.4), (-1.7, 1.7)),
    )
    result = plan_reach_sequence(
        args.planner,
        robot=robot,
        start=start,
        goals_xyz=goals_xyz,
        obstacles=obstacles,
        config=config,
    )
    trajectory = result.trajectory
    segment_ends = result.segment_ends
    terminal_errors = result.terminal_errors
    clearances = result.clearances
    iterations = result.iterations
    planning_times = result.planning_times
    messages = result.messages
    success = result.success
    planner_spec = get_planner(args.planner)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "moma_rrt_connect.npz"
    html_path = args.output_dir / "moma_rrt_connect.html"
    np.savez(
        npz_path,
        trajectory=trajectory,
        start=start,
        goals_xyz=goals_xyz,
        segment_ends=np.array(segment_ends, dtype=int),
        success=success,
        terminal_errors=np.array(terminal_errors, dtype=float),
        clearances=np.array(clearances, dtype=float),
        iterations=np.array(iterations, dtype=int),
        planning_times=np.array(planning_times, dtype=float),
        messages=np.array(messages, dtype=object),
        planner=np.array(args.planner),
    )

    visual_trajectory, segment_ends = densify_trajectory_for_visualization(
        trajectory,
        segment_ends,
        max_state_step=0.08,
    )
    export_reach_sequence_html(
        robot,
        visual_trajectory,
        goals_xyz,
        segment_ends,
        obstacles,
        html_path,
        metadata={
            "planner": args.planner,
            "plannerDisplayName": planner_spec.display_name,
            "success": success,
            "seed": args.seed,
            "obstacleCount": len(obstacles),
            "goalCount": len(goals_xyz),
            "terminalErrors": terminal_errors,
            "segmentClearances": clearances,
            "iterations": iterations,
            "planningTimes": planning_times,
            "messages": messages,
        },
    )

    print("MoMa/Piper whole-body planning demo")
    print(f"  planner:          {args.planner} ({planner_spec.display_name})")
    print(f"  robot state dim:  {robot.state_dim}")
    print(f"  goals:            {len(goals_xyz)}")
    print(f"  success:          {success}")
    print(f"  trajectory steps: {len(trajectory)}")
    total_planning_time = sum(planning_times)
    print(f"  planning time:    {total_planning_time:.3f} s")
    for i, (err, clearance, iters, plan_time, message) in enumerate(
        zip(terminal_errors, clearances, iterations, planning_times, messages),
        start=1,
    ):
        print(
            f"  goal {i}: error={err:.4f} m, clearance={clearance:.4f} m, "
            f"iterations={iters}, planning_time={plan_time:.3f} s, message={message}"
        )
    print(f"  saved npz:        {npz_path}")
    print(f"  saved html:       {html_path}")


def build_moma_demo_scene(seed: int = 3, obstacle_count: int = 24) -> tuple[np.ndarray, np.ndarray, list[Obstacle3D]]:
    robot = MomaPiperMobileManipulator3D()
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
    return start, goals_xyz, obstacles


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


def _distance_to_start_goal_corridor(point: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    segment = goal_xy - start_xy
    denom = float(np.dot(segment, segment))
    if denom < 1e-9:
        return float(np.linalg.norm(point - start_xy))
    t = np.clip(float(np.dot(point - start_xy, segment) / denom), 0.0, 1.0)
    closest = start_xy + t * segment
    return float(np.linalg.norm(point - closest))


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


if __name__ == "__main__":
    main()
