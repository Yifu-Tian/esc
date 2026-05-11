#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mm_flow.three_d.metrics import compute_plan_metrics
from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D
from mm_flow.three_d.planners import available_planners, get_planner, plan_problem
from mm_flow.three_d.problems import build_moma_reach_sequence_problem
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.run_io import save_reach_sequence_run
from mm_flow.three_d.serialization import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark registered whole-body planners on MoMa reach scenes.")
    parser.add_argument("--planners", nargs="+", choices=available_planners(), default=["rrt_connect"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3])
    parser.add_argument("--num-scenes", type=int, default=None, help="Optional number of sequential seeds starting at the first seed.")
    parser.add_argument("--obstacle-count", type=int, default=24)
    parser.add_argument("--max-iterations", type=int, default=9000)
    parser.add_argument("--goal-samples", type=int, default=180)
    parser.add_argument("--step-size", type=float, default=0.18)
    parser.add_argument("--edge-resolution", type=float, default=0.06)
    parser.add_argument("--clearance-margin", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--no-html", action="store_true", help="Skip per-run HTML export for faster benchmark sweeps.")
    args = parser.parse_args()

    seeds = args.seeds
    if args.num_scenes is not None:
        start_seed = seeds[0] if seeds else 0
        seeds = list(range(start_seed, start_seed + args.num_scenes))

    robot = MomaPiperMobileManipulator3D()
    rows = []
    for planner_name in args.planners:
        planner_spec = get_planner(planner_name)
        for seed in seeds:
            bounds_xy = ((-2.4, 2.4), (-1.7, 1.7))
            problem = build_moma_reach_sequence_problem(
                robot=robot,
                seed=seed,
                obstacle_count=args.obstacle_count,
                bounds_xy=bounds_xy,
            )
            config = RRTConnectConfig(
                max_iterations=args.max_iterations,
                step_size=args.step_size,
                edge_resolution=args.edge_resolution,
                goal_sample_count=args.goal_samples,
                shortcut_attempts=80,
                rng_seed=seed,
                clearance_margin=args.clearance_margin,
                bounds_xy=bounds_xy,
            )
            result = plan_problem(planner_name, robot, problem, config)
            metrics = compute_plan_metrics(problem, result)
            run_id = f"{planner_name}_seed{seed:04d}"
            run_dir = args.output_dir / run_id
            if args.no_html:
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_reach_sequence_run(
                    robot,
                    problem,
                    result,
                    run_dir,
                    planner_display_name=planner_spec.display_name,
                    config=config,
                )
            row = {
                "run_id": run_id,
                "planner": planner_name,
                "seed": seed,
                "success": result.success,
                "failure_reason": result.failure_reason,
                "time": metrics.total_planning_time,
                "min_clearance": metrics.min_clearance,
                "mean_clearance": metrics.mean_clearance,
                "path_length": metrics.path_length,
                "base_path_length": metrics.base_path_length,
                "joint_motion_length": metrics.joint_motion_length,
                "ee_path_length": metrics.end_effector_path_length,
                "smoothness": metrics.smoothness,
                "iterations": metrics.iterations,
            }
            rows.append(row)
            print(
                f"{planner_name:18s} seed={seed:4d} success={result.success} "
                f"time={metrics.total_planning_time:.3f}s clearance={metrics.min_clearance:.4f} "
                f"path={metrics.path_length:.3f} smooth={metrics.smoothness:.3f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(to_jsonable(rows), indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved summary csv:  {csv_path}")
    print(f"saved summary json: {json_path}")


if __name__ == "__main__":
    main()
