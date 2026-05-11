#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D
from mm_flow.three_d.planners import RecedingRRTConnectConfig, available_planners, get_planner, plan_problem
from mm_flow.three_d.problems import build_moma_reach_sequence_problem
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.run_io import save_reach_sequence_run


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
    parser.add_argument(
        "--problem-variant",
        choices=["standard", "narrow_passage", "arm_obstacle", "base_required"],
        default="standard",
    )
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument("--execute-steps", type=int, default=6)
    parser.add_argument("--max-cycles-per-goal", type=int, default=8)
    parser.add_argument("--disturbance-std", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moma_rrt_connect"))
    args = parser.parse_args()

    robot = MomaPiperMobileManipulator3D()
    bounds_xy = ((-2.4, 2.4), (-1.7, 1.7))
    problem = build_moma_reach_sequence_problem(
        robot=robot,
        seed=args.seed,
        obstacle_count=args.obstacle_count,
        bounds_xy=bounds_xy,
        variant=args.problem_variant,
    )
    rrt_config = RRTConnectConfig(
        max_iterations=args.max_iterations,
        step_size=args.step_size,
        edge_resolution=args.edge_resolution,
        goal_sample_count=args.goal_samples,
        shortcut_attempts=80,
        rng_seed=args.seed,
        clearance_margin=args.clearance_margin,
        bounds_xy=bounds_xy,
    )
    if args.planner == "rrt_connect_receding":
        config = RecedingRRTConnectConfig(
            rrt=rrt_config,
            horizon_steps=args.horizon_steps,
            execute_steps=args.execute_steps,
            max_cycles_per_goal=args.max_cycles_per_goal,
            disturbance_std=args.disturbance_std,
            follower_seed=args.seed,
        )
    else:
        config = rrt_config
    result = plan_problem(
        args.planner,
        robot=robot,
        problem=problem,
        config=config,
    )
    terminal_errors = result.terminal_errors
    clearances = result.clearances
    iterations = result.iterations
    planning_times = result.planning_times
    messages = result.messages
    success = result.success
    planner_spec = get_planner(args.planner)

    output_paths = save_reach_sequence_run(
        robot,
        problem,
        result,
        args.output_dir,
        planner_display_name=planner_spec.display_name,
        config=config,
    )

    print("MoMa/Piper whole-body planning demo")
    print(f"  planner:          {args.planner} ({planner_spec.display_name})")
    print(f"  problem:          {problem.name}")
    print(f"  robot state dim:  {robot.state_dim}")
    print(f"  goals:            {problem.goal_count}")
    print(f"  success:          {success}")
    print(f"  trajectory steps: {result.trajectory_steps}")
    print(f"  planning time:    {result.total_planning_time:.3f} s")
    for i, (err, clearance, iters, plan_time, message) in enumerate(
        zip(terminal_errors, clearances, iterations, planning_times, messages),
        start=1,
    ):
        print(
            f"  goal {i}: error={err:.4f} m, clearance={clearance:.4f} m, "
            f"iterations={iters}, planning_time={plan_time:.3f} s, message={message}"
        )
    print(f"  saved trajectory: {output_paths['trajectory']}")
    print(f"  saved html:       {output_paths['html']}")
    print(f"  saved result:     {output_paths['result']}")


if __name__ == "__main__":
    main()
