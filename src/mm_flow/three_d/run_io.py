from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from mm_flow.three_d.html_viewer import export_reach_sequence_html
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.metrics import compute_plan_metrics
from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.results import ReachSequencePlanResult
from mm_flow.three_d.serialization import obstacle_to_dict, to_jsonable
from mm_flow.three_d.tasks import densify_trajectory_for_visualization


def save_reach_sequence_run(
    robot: SimpleMobileManipulator3D,
    problem: ReachSequenceProblem,
    result: ReachSequencePlanResult,
    output_dir: Path,
    planner_display_name: str,
    config: object | None = None,
    html_name: str = "animation.html",
    npz_name: str = "trajectory.npz",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / npz_name
    html_path = output_dir / html_name
    config_path = output_dir / "config.json"
    scene_path = output_dir / "scene.json"
    result_path = output_dir / "result.json"
    metrics = compute_plan_metrics(problem, result)

    np.savez(
        npz_path,
        trajectory=result.trajectory,
        start=problem.start,
        goals_xyz=problem.goals_xyz,
        segment_ends=np.array(result.segment_ends, dtype=int),
        success=result.success,
        terminal_errors=np.array(result.terminal_errors, dtype=float),
        clearances=np.array(result.clearances, dtype=float),
        iterations=np.array(result.iterations, dtype=int),
        planning_times=np.array(result.planning_times, dtype=float),
        messages=np.array(result.messages, dtype=object),
        planner=np.array(result.planner_name),
        problem=np.array(problem.name),
        seed=np.array(problem.seed, dtype=int),
    )

    _write_json(
        config_path,
        {
            "planner": result.planner_name,
            "planner_display_name": planner_display_name,
            "config": _config_to_dict(config),
        },
    )
    _write_json(
        scene_path,
        {
            "name": problem.name,
            "task_type": problem.task_type,
            "seed": problem.seed,
            "bounds_xy": problem.bounds_xy,
            "start": problem.start,
            "goals_xyz": problem.goals_xyz,
            "obstacles": [obstacle_to_dict(obstacle) for obstacle in problem.obstacles],
            "metadata": problem.metadata,
        },
    )
    _write_json(
        result_path,
        {
            "planner": result.planner_name,
            "success": result.success,
            "failure_reason": result.failure_reason,
            "trajectory_steps": result.trajectory_steps,
            "segment_ends": result.segment_ends,
            "terminal_errors": result.terminal_errors,
            "clearances": result.clearances,
            "iterations": result.iterations,
            "planning_times": result.planning_times,
            "total_planning_time": result.total_planning_time,
            "messages": result.messages,
            "metadata": result.metadata,
            "metrics": asdict(metrics),
        },
    )

    visualization_step = 0.08
    visual_trajectory, visual_segment_ends = densify_trajectory_for_visualization(
        result.trajectory,
        result.segment_ends,
        max_state_step=visualization_step,
    )
    visual_index_map = _visual_index_map(result.trajectory, max_state_step=visualization_step)
    export_reach_sequence_html(
        robot,
        visual_trajectory,
        problem.goals_xyz,
        visual_segment_ends,
        problem.obstacles,
        html_path,
        metadata={
            "planner": result.planner_name,
            "plannerDisplayName": planner_display_name,
            "success": result.success,
            "problem": problem.name,
            "seed": problem.seed,
            "obstacleCount": problem.obstacle_count,
            "goalCount": problem.goal_count,
            "terminalErrors": result.terminal_errors,
            "segmentClearances": result.clearances,
            "iterations": result.iterations,
            "planningTimes": result.planning_times,
            "messages": result.messages,
            "metrics": asdict(metrics),
            "pathLength": metrics.path_length,
            "basePathLength": metrics.base_path_length,
            "jointMotionLength": metrics.joint_motion_length,
            "eePathLength": metrics.end_effector_path_length,
            "smoothness": metrics.smoothness,
            "collisionCheckCount": metrics.collision_check_count,
            "perGoalPathLength": metrics.per_goal_path_length,
            "perGoalBasePathLength": metrics.per_goal_base_path_length,
            "perGoalJointMotionLength": metrics.per_goal_joint_motion_length,
            "perGoalEePathLength": metrics.per_goal_end_effector_path_length,
            "perGoalMinClearance": metrics.per_goal_min_clearance,
            "resultMetadata": result.metadata,
            "recedingCycles": _map_receding_cycles_to_visual_frames(
                result.metadata.get("receding_cycles", []),
                visual_index_map,
            ),
        },
    )
    return {
        "config": config_path,
        "scene": scene_path,
        "result": result_path,
        "trajectory": npz_path,
        "html": html_path,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def _config_to_dict(config: object | None) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


def _visual_index_map(trajectory: np.ndarray, max_state_step: float) -> dict[int, int]:
    index_map = {0: 0}
    dense_index = 0
    for i in range(len(trajectory) - 1):
        delta = trajectory[i + 1] - trajectory[i]
        steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_state_step)))
        dense_index += steps
        index_map[i + 1] = dense_index
    return index_map


def _map_receding_cycles_to_visual_frames(
    cycles: object,
    visual_index_map: dict[int, int],
) -> list[dict[str, Any]]:
    if not isinstance(cycles, list):
        return []
    mapped = []
    last_visual_index = max(visual_index_map.values()) if visual_index_map else 0
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        record = dict(cycle)
        frame_start = _safe_int(record.get("frame_start"), 0)
        frame_end = _safe_int(record.get("frame_end"), frame_start)
        record["planning_frame_start"] = frame_start
        record["planning_frame_end"] = frame_end
        record["frame_start"] = visual_index_map.get(frame_start, last_visual_index)
        record["frame_end"] = visual_index_map.get(frame_end, last_visual_index)
        mapped.append(record)
    return mapped


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
