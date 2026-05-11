#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mm_flow.three_d.html_viewer import export_reach_sequence_html
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.rrt_connect import RRTConnectConfig, RRTPlanResult, plan_rrt_connect_to_goal
from mm_flow.three_d.tasks import build_reach_sequence_scene_3d, densify_trajectory_for_visualization


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a reproducible RRT-Connect failure under a tight online replanning budget."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--goal-index", type=int, default=0)
    parser.add_argument("--clearance-margin", type=float, default=0.02)
    parser.add_argument("--low-max-iterations", type=int, default=250)
    parser.add_argument("--low-goal-samples", type=int, default=20)
    parser.add_argument("--reference-max-iterations", type=int, default=8000)
    parser.add_argument("--reference-goal-samples", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rrt_connect_failure_validation"))
    args = parser.parse_args()

    robot = SimpleMobileManipulator3D()
    start, goals_xyz, obstacles = build_reach_sequence_scene_3d(
        seed=args.seed,
        random_obstacle_count=args.obstacle_count,
    )
    goal_xyz = goals_xyz[args.goal_index]

    low_config = RRTConnectConfig(
        max_iterations=args.low_max_iterations,
        goal_sample_count=args.low_goal_samples,
        shortcut_attempts=0,
        rng_seed=args.seed,
        clearance_margin=args.clearance_margin,
    )
    reference_config = RRTConnectConfig(
        max_iterations=args.reference_max_iterations,
        goal_sample_count=args.reference_goal_samples,
        shortcut_attempts=80,
        rng_seed=args.seed,
        clearance_margin=args.clearance_margin,
    )

    low_result, low_time = _timed_plan(robot, start, goal_xyz, obstacles, low_config)
    reference_result, reference_time = _timed_plan(robot, start, goal_xyz, obstacles, reference_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    npz_path = args.output_dir / "reference_solution.npz"
    html_path = args.output_dir / "reference_solution.html"

    summary = {
        "scenario": {
            "seed": args.seed,
            "obstacle_count": args.obstacle_count,
            "goal_index": args.goal_index,
            "goal_xyz": goal_xyz.tolist(),
            "clearance_margin": args.clearance_margin,
        },
        "low_budget": _result_summary(low_result, low_time, low_config),
        "reference_budget": _result_summary(reference_result, reference_time, reference_config),
        "interpretation": (
            "The same planning problem is solvable with a larger budget, but fails under a tight "
            "online replanning budget. This motivates amortized plan-to-plan generation and "
            "receding-horizon reuse of the previous whole-body trajectory."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.savez(
        npz_path,
        start=start,
        goal_xyz=goal_xyz,
        obstacles=np.array([str(obstacle) for obstacle in obstacles], dtype=object),
        low_success=low_result.success,
        low_terminal_error=low_result.terminal_error,
        low_min_clearance=low_result.min_clearance,
        reference_trajectory=reference_result.trajectory,
        reference_success=reference_result.success,
        reference_terminal_error=reference_result.terminal_error,
        reference_min_clearance=reference_result.min_clearance,
    )

    if reference_result.success:
        visual_trajectory, segment_ends = densify_trajectory_for_visualization(
            reference_result.trajectory,
            [len(reference_result.trajectory) - 1],
        )
        export_reach_sequence_html(
            robot,
            visual_trajectory,
            goal_xyz[None, :],
            segment_ends,
            obstacles,
            html_path,
            metadata={
                "planner": "rrt_connect_reference",
                "success": reference_result.success,
                "seed": args.seed,
                "obstacleCount": args.obstacle_count,
                "lowBudgetSuccess": low_result.success,
                "lowBudgetMessage": low_result.message,
                "lowBudgetTimeSec": low_time,
                "referenceTimeSec": reference_time,
                "clearanceMargin": args.clearance_margin,
            },
        )

    print("RRT-Connect failure validation")
    print(f"  scene:             seed={args.seed}, obstacles={args.obstacle_count}, goal={args.goal_index + 1}")
    print(f"  clearance margin:  {args.clearance_margin:.3f} m")
    print(
        "  low budget:        "
        f"success={low_result.success}, message={low_result.message}, "
        f"time={low_time:.3f}s, error={low_result.terminal_error:.4f}m"
    )
    print(
        "  reference budget:  "
        f"success={reference_result.success}, message={reference_result.message}, "
        f"time={reference_time:.3f}s, error={reference_result.terminal_error:.4f}m, "
        f"clearance={reference_result.min_clearance:.4f}m"
    )
    print(f"  saved summary:     {summary_path}")
    print(f"  saved npz:         {npz_path}")
    if reference_result.success:
        print(f"  saved html:        {html_path}")


def _timed_plan(
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goal_xyz: np.ndarray,
    obstacles: list,
    config: RRTConnectConfig,
) -> tuple[RRTPlanResult, float]:
    start_time = time.perf_counter()
    result = plan_rrt_connect_to_goal(robot, start, goal_xyz, obstacles, config)
    return result, time.perf_counter() - start_time


def _result_summary(result: RRTPlanResult, elapsed_sec: float, config: RRTConnectConfig) -> dict:
    return {
        "success": result.success,
        "message": result.message,
        "elapsed_sec": elapsed_sec,
        "terminal_error": result.terminal_error,
        "min_clearance": result.min_clearance,
        "iterations": result.iterations,
        "max_iterations": config.max_iterations,
        "goal_sample_count": config.goal_sample_count,
        "shortcut_attempts": config.shortcut_attempts,
    }


if __name__ == "__main__":
    main()
