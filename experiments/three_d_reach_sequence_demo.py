#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mm_flow.three_d.html_viewer import export_reach_sequence_html
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.tasks import (
    build_reach_sequence_scene_3d,
    densify_trajectory_for_visualization,
    plan_reach_sequence_3d,
)
from mm_flow.three_d.visualization import animate_reach_sequence_3d, plot_reach_sequence_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified 3D mobile manipulator sequential reaching demo.")
    parser.add_argument("--segment-horizon", type=int, default=42)
    parser.add_argument("--planner", choices=["heuristic", "rrt_connect"], default="rrt_connect")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obstacle-count", type=int, default=12)
    parser.add_argument("--rrt-max-iterations", type=int, default=6000)
    parser.add_argument("--rrt-step-size", type=float, default=0.22)
    parser.add_argument("--rrt-edge-resolution", type=float, default=0.08)
    parser.add_argument("--rrt-goal-samples", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/three_d_reach_sequence"))
    args = parser.parse_args()

    robot = SimpleMobileManipulator3D()
    start, goals_xyz, obstacles = build_reach_sequence_scene_3d(
        seed=args.seed,
        random_obstacle_count=args.obstacle_count,
    )
    trajectory, segment_ends, terminal_errors, clearances, success = plan_reach_sequence_3d(
        robot=robot,
        start=start,
        goals_xyz=goals_xyz,
        obstacles=obstacles,
        segment_horizon=args.segment_horizon,
        planner=args.planner,
        rrt_config=RRTConnectConfig(
            max_iterations=args.rrt_max_iterations,
            step_size=args.rrt_step_size,
            edge_resolution=args.rrt_edge_resolution,
            goal_sample_count=args.rrt_goal_samples,
            rng_seed=args.seed,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "reach_sequence_3d.npz"
    png_path = args.output_dir / "reach_sequence_3d.png"
    gif_path = args.output_dir / "reach_sequence_3d.gif"
    html_path = args.output_dir / "reach_sequence_3d.html"
    np.savez(
        npz_path,
        trajectory=trajectory,
        start=start,
        goals_xyz=goals_xyz,
        segment_ends=np.array(segment_ends, dtype=int),
        terminal_errors=np.array(terminal_errors, dtype=float),
        clearances=np.array(clearances, dtype=float),
        success=success,
        seed=args.seed,
        obstacle_count=args.obstacle_count,
        planner=args.planner,
    )
    visual_trajectory, visual_segment_ends = densify_trajectory_for_visualization(trajectory, segment_ends)
    plot_reach_sequence_3d(robot, visual_trajectory, goals_xyz, visual_segment_ends, obstacles, png_path)
    animate_reach_sequence_3d(robot, visual_trajectory, goals_xyz, visual_segment_ends, obstacles, gif_path)
    export_reach_sequence_html(
        robot,
        visual_trajectory,
        goals_xyz,
        visual_segment_ends,
        obstacles,
        html_path,
        metadata={
            "planner": args.planner,
            "success": success,
            "seed": args.seed,
            "obstacleCount": args.obstacle_count,
            "rawTrajectorySteps": len(trajectory),
            "visualizationSteps": len(visual_trajectory),
            "terminalErrors": terminal_errors,
            "segmentClearances": clearances,
            "rrtMaxIterations": args.rrt_max_iterations,
            "rrtStepSize": args.rrt_step_size,
            "rrtEdgeResolution": args.rrt_edge_resolution,
            "rrtGoalSamples": args.rrt_goal_samples,
        },
    )

    print("3D reach-sequence demo")
    print(f"  planner:          {args.planner}")
    print(f"  success:          {success}")
    print(f"  goals:            {len(goals_xyz)}")
    print(f"  trajectory steps: {len(trajectory)}")
    for i, (err, clearance) in enumerate(zip(terminal_errors, clearances), start=1):
        print(f"  goal {i}: error={err:.4f} m, min_clearance={clearance:.4f} m")
    print(f"  saved npz:        {npz_path}")
    print(f"  saved figure:     {png_path}")
    print(f"  saved animation:  {gif_path}")
    print(f"  saved html:       {html_path}")


if __name__ == "__main__":
    main()
