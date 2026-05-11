from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_flow.three_d.collision import state_clearance
from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.results import ReachSequencePlanResult


@dataclass(frozen=True)
class PlanMetrics:
    success: bool
    total_planning_time: float
    per_goal_planning_time: list[float]
    per_goal_error: list[float]
    per_goal_path_length: list[float]
    per_goal_base_path_length: list[float]
    per_goal_joint_motion_length: list[float]
    per_goal_end_effector_path_length: list[float]
    per_goal_min_clearance: list[float]
    min_clearance: float
    mean_clearance: float
    path_length: float
    base_path_length: float
    joint_motion_length: float
    end_effector_path_length: float
    smoothness: float
    velocity_violation: float
    acceleration_violation: float
    collision_check_count: int
    per_goal_collision_checks: list[int]
    replanning_count: int
    iterations: int


def compute_plan_metrics(
    problem: ReachSequenceProblem,
    result: ReachSequencePlanResult,
    max_state_step: float = 0.25,
    max_acc_step: float = 0.35,
) -> PlanMetrics:
    trajectory = np.asarray(result.trajectory, dtype=float)
    if len(trajectory) < 2:
        path_length = 0.0
        base_path_length = 0.0
        joint_motion_length = 0.0
        smoothness = 0.0
        velocity_violation = 0.0
        acceleration_violation = 0.0
    else:
        delta = np.diff(trajectory, axis=0)
        delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
        if trajectory.shape[1] > 3:
            delta[:, 3:] = np.arctan2(np.sin(delta[:, 3:]), np.cos(delta[:, 3:]))
        step_norms = np.linalg.norm(delta, axis=1)
        path_length = float(np.sum(step_norms))
        base_path_length = float(np.sum(np.linalg.norm(delta[:, :2], axis=1)))
        joint_motion_length = float(np.sum(np.linalg.norm(delta[:, 3:], axis=1))) if trajectory.shape[1] > 3 else 0.0
        velocity_violation = float(np.maximum(step_norms - max_state_step, 0.0).sum())

        if len(delta) >= 2:
            accel = np.diff(delta, axis=0)
            accel_norms = np.linalg.norm(accel, axis=1)
            smoothness = float(np.sum(accel_norms**2))
            acceleration_violation = float(np.maximum(accel_norms - max_acc_step, 0.0).sum())
        else:
            smoothness = 0.0
            acceleration_violation = 0.0

    ee_path = np.array([problem.robot.end_effector(state) for state in trajectory], dtype=float)
    if len(ee_path) < 2:
        end_effector_path_length = 0.0
    else:
        end_effector_path_length = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)))

    clearances = [state_clearance(problem.robot, state, problem.obstacles) for state in trajectory]
    min_clearance = float(np.min(clearances)) if clearances else float("nan")
    mean_clearance = float(np.mean(clearances)) if clearances else float("nan")
    segment_metrics = _compute_segment_metrics(problem, trajectory, result.segment_ends)

    return PlanMetrics(
        success=result.success,
        total_planning_time=result.total_planning_time,
        per_goal_planning_time=list(result.planning_times),
        per_goal_error=list(result.terminal_errors),
        per_goal_path_length=segment_metrics["path_length"],
        per_goal_base_path_length=segment_metrics["base_path_length"],
        per_goal_joint_motion_length=segment_metrics["joint_motion_length"],
        per_goal_end_effector_path_length=segment_metrics["end_effector_path_length"],
        per_goal_min_clearance=segment_metrics["min_clearance"],
        min_clearance=min_clearance,
        mean_clearance=mean_clearance,
        path_length=path_length,
        base_path_length=base_path_length,
        joint_motion_length=joint_motion_length,
        end_effector_path_length=end_effector_path_length,
        smoothness=smoothness,
        velocity_violation=velocity_violation,
        acceleration_violation=acceleration_violation,
        collision_check_count=int(result.metadata.get("collision_check_count", 0)),
        per_goal_collision_checks=list(result.metadata.get("per_goal_collision_checks", [])),
        replanning_count=int(result.metadata.get("replanning_count", 0)),
        iterations=result.iteration_count,
    )


def _compute_segment_metrics(
    problem: ReachSequenceProblem,
    trajectory: np.ndarray,
    segment_ends: list[int],
) -> dict[str, list[float]]:
    metrics = {
        "path_length": [],
        "base_path_length": [],
        "joint_motion_length": [],
        "end_effector_path_length": [],
        "min_clearance": [],
    }
    start_idx = 0
    for end_idx in segment_ends:
        end_idx = max(start_idx, min(int(end_idx), len(trajectory) - 1))
        segment = trajectory[start_idx : end_idx + 1]
        if len(segment) < 2:
            delta = np.zeros((0, trajectory.shape[1]), dtype=float)
        else:
            delta = np.diff(segment, axis=0)
            delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
            if segment.shape[1] > 3:
                delta[:, 3:] = np.arctan2(np.sin(delta[:, 3:]), np.cos(delta[:, 3:]))
        metrics["path_length"].append(float(np.sum(np.linalg.norm(delta, axis=1))) if len(delta) else 0.0)
        metrics["base_path_length"].append(float(np.sum(np.linalg.norm(delta[:, :2], axis=1))) if len(delta) else 0.0)
        metrics["joint_motion_length"].append(float(np.sum(np.linalg.norm(delta[:, 3:], axis=1))) if len(delta) and segment.shape[1] > 3 else 0.0)
        ee_path = np.array([problem.robot.end_effector(state) for state in segment], dtype=float)
        metrics["end_effector_path_length"].append(
            float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1))) if len(ee_path) > 1 else 0.0
        )
        segment_clearances = [state_clearance(problem.robot, state, problem.obstacles) for state in segment]
        metrics["min_clearance"].append(float(np.min(segment_clearances)) if segment_clearances else float("nan"))
        start_idx = end_idx + 1
    return metrics
