from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from mm_flow.three_d.collision import trajectory_min_clearance
from mm_flow.three_d.execution import FirstOrderTrajectoryFollower, TrajectoryFollower
from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.results import ReachSequencePlanResult
from mm_flow.three_d.rrt_connect import RRTConnectConfig, plan_rrt_connect_to_goal


@dataclass
class Observation:
    current_state: np.ndarray
    obstacles: object
    goals_xyz: np.ndarray
    previous_plan: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)


@dataclass
class PlanBuffer:
    trajectory: np.ndarray | None = None

    def shifted(self, executed_steps: int) -> np.ndarray | None:
        if self.trajectory is None:
            return None
        if executed_steps >= len(self.trajectory):
            return self.trajectory[-1:]
        return self.trajectory[executed_steps:]


class RecedingHorizonRunner:
    def __init__(
        self,
        horizon_steps: int,
        execute_steps: int,
        max_cycles_per_goal: int,
        follower: TrajectoryFollower | None = None,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.execute_steps = execute_steps
        self.max_cycles_per_goal = max_cycles_per_goal
        self.follower = follower or FirstOrderTrajectoryFollower(alpha=1.0)

    def run(self, problem: ReachSequenceProblem, planner_name: str, config: object) -> ReachSequencePlanResult:
        if planner_name != "rrt_connect":
            raise NotImplementedError(f"Receding runner currently supports rrt_connect, got {planner_name!r}.")
        if not isinstance(config, RRTConnectConfig):
            raise TypeError("Receding runner expects RRTConnectConfig.")
        return plan_receding_rrt_connect(problem, config, self)


def plan_receding_rrt_connect(
    problem: ReachSequenceProblem,
    config: RRTConnectConfig,
    runner: RecedingHorizonRunner,
) -> ReachSequencePlanResult:
    robot = problem.robot
    current = np.asarray(problem.start, dtype=float).copy()
    pieces = [current[None, :]]
    segment_ends: list[int] = []
    terminal_errors: list[float] = []
    clearances: list[float] = []
    iterations: list[int] = []
    planning_times: list[float] = []
    messages: list[str] = []
    cycle_times: list[float] = []
    tracking_errors: list[float] = []
    discontinuities: list[float] = []
    cycle_records: list[dict[str, object]] = []
    collision_check_counts: list[int] = []
    failed_replans = 0
    plan_buffer = PlanBuffer()
    success = True

    for goal_index, goal_xyz in enumerate(problem.goals_xyz):
        reached = False
        goal_iterations = 0
        goal_plan_time = 0.0
        goal_messages: list[str] = []
        min_goal_clearance = float("inf")

        for cycle in range(runner.max_cycles_per_goal):
            frame_before = sum(len(piece) for piece in pieces) - 1
            cycle_config = RRTConnectConfig(
                max_iterations=config.max_iterations,
                step_size=config.step_size,
                edge_resolution=config.edge_resolution,
                goal_tolerance=config.goal_tolerance,
                goal_sample_count=config.goal_sample_count,
                shortcut_attempts=config.shortcut_attempts,
                rng_seed=config.rng_seed + goal_index * 1009 + cycle * 37,
                clearance_margin=config.clearance_margin,
                bounds_xy=config.bounds_xy,
            )
            start_time = time.perf_counter()
            plan = plan_rrt_connect_to_goal(robot, current, goal_xyz, problem.obstacles, cycle_config)
            plan_time = time.perf_counter() - start_time
            cycle_times.append(plan_time)
            goal_plan_time += plan_time
            goal_iterations += plan.iterations
            goal_messages.append(plan.message)
            collision_check_counts.append(plan.collision_checks)

            previous_reference = plan_buffer.shifted(runner.execute_steps)
            discontinuity = (
                float(np.linalg.norm(plan.trajectory[0] - previous_reference[0]))
                if previous_reference is not None and len(previous_reference) > 0
                else 0.0
            )
            discontinuities.append(discontinuity)
            plan_buffer.trajectory = plan.trajectory

            if not plan.success:
                failed_replans += 1
                min_goal_clearance = min(min_goal_clearance, plan.min_clearance)
                cycle_records.append(
                    _cycle_record(
                        cycle_index=len(cycle_records) + 1,
                        goal_index=goal_index,
                        local_cycle=cycle + 1,
                        frame_start=frame_before,
                        frame_end=frame_before,
                        plan=plan,
                        planning_time=plan_time,
                        executed_steps=0,
                        tracking_errors=np.array([], dtype=float),
                        discontinuity=discontinuity,
                        state_error_after_execute=float(np.linalg.norm(robot.end_effector(current) - goal_xyz)),
                    )
                )
                break

            horizon = plan.trajectory[: max(1, min(runner.horizon_steps, len(plan.trajectory)))]
            executed_len = max(1, min(runner.execute_steps, len(horizon)))
            execution = runner.follower.execute(horizon[1 : executed_len + 1], current)
            executed = execution.executed_states
            tracking_errors.extend(execution.tracking_errors.tolist())
            if len(executed) > 0:
                pieces.append(executed)
                current = executed[-1].copy()
            min_goal_clearance = min(min_goal_clearance, trajectory_min_clearance(robot, np.vstack([current[None, :], horizon]), problem.obstacles))

            terminal_error = float(np.linalg.norm(robot.end_effector(current) - goal_xyz))
            cycle_records.append(
                _cycle_record(
                    cycle_index=len(cycle_records) + 1,
                    goal_index=goal_index,
                    local_cycle=cycle + 1,
                    frame_start=frame_before + 1 if len(executed) > 0 else frame_before,
                    frame_end=sum(len(piece) for piece in pieces) - 1,
                    plan=plan,
                    planning_time=plan_time,
                    executed_steps=len(executed),
                    tracking_errors=execution.tracking_errors,
                    discontinuity=discontinuity,
                    state_error_after_execute=terminal_error,
                )
            )
            if terminal_error <= config.goal_tolerance:
                reached = True
                break

        terminal_error = float(np.linalg.norm(robot.end_effector(current) - goal_xyz))
        terminal_errors.append(terminal_error)
        clearances.append(float(min_goal_clearance if np.isfinite(min_goal_clearance) else 0.0))
        iterations.append(goal_iterations)
        planning_times.append(goal_plan_time)
        messages.append("reached" if reached else "; ".join(dict.fromkeys(goal_messages)) or "not reached")
        segment_ends.append(sum(len(piece) for piece in pieces) - 1)
        success = success and reached
        if not reached:
            break

    while len(segment_ends) < len(problem.goals_xyz):
        terminal_errors.append(float(np.linalg.norm(robot.end_effector(current) - problem.goals_xyz[len(segment_ends)])))
        clearances.append(float(trajectory_min_clearance(robot, current[None, :], problem.obstacles)))
        iterations.append(0)
        planning_times.append(0.0)
        messages.append("skipped after failed replan")
        segment_ends.append(sum(len(piece) for piece in pieces) - 1)
        success = False

    trajectory = np.vstack(pieces)
    trajectory[:, 2] = np.unwrap(trajectory[:, 2])
    return ReachSequencePlanResult(
        trajectory=trajectory,
        segment_ends=segment_ends,
        terminal_errors=terminal_errors,
        clearances=clearances,
        iterations=iterations,
        planning_times=planning_times,
        messages=messages,
        success=bool(success),
        planner_name="rrt_connect_receding",
        failure_reason="" if success else "; ".join(dict.fromkeys(message for message in messages if message != "reached")),
        metadata={
            "base_planner": "rrt_connect",
            "collision_checker": "proxy",
            "task_type": problem.task_type,
            "replanning_count": len(cycle_times),
            "failed_replans": failed_replans,
            "collision_check_count": int(sum(collision_check_counts)),
            "per_replan_collision_checks": collision_check_counts,
            "cycle_planning_times": cycle_times,
            "receding_cycles": cycle_records,
            "mean_tracking_error": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
            "max_tracking_error": float(np.max(tracking_errors)) if tracking_errors else 0.0,
            "mean_plan_discontinuity": float(np.mean(discontinuities)) if discontinuities else 0.0,
            "max_plan_discontinuity": float(np.max(discontinuities)) if discontinuities else 0.0,
            "horizon_steps": runner.horizon_steps,
            "execute_steps": runner.execute_steps,
            "max_cycles_per_goal": runner.max_cycles_per_goal,
        },
    )


def _cycle_record(
    cycle_index: int,
    goal_index: int,
    local_cycle: int,
    frame_start: int,
    frame_end: int,
    plan: object,
    planning_time: float,
    executed_steps: int,
    tracking_errors: np.ndarray,
    discontinuity: float,
    state_error_after_execute: float,
) -> dict[str, object]:
    return {
        "cycle": int(cycle_index),
        "goal_index": int(goal_index),
        "goal_label": f"g{goal_index + 1}",
        "local_cycle": int(local_cycle),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "planning_time": float(planning_time),
        "iterations": int(getattr(plan, "iterations", 0)),
        "planner_message": str(getattr(plan, "message", "")),
        "plan_success": bool(getattr(plan, "success", False)),
        "terminal_error_before_execute": float(getattr(plan, "terminal_error", 0.0)),
        "clearance": float(getattr(plan, "min_clearance", 0.0)),
        "executed_steps": int(executed_steps),
        "tracking_error_mean": float(np.mean(tracking_errors)) if len(tracking_errors) else 0.0,
        "tracking_error_max": float(np.max(tracking_errors)) if len(tracking_errors) else 0.0,
        "plan_discontinuity": float(discontinuity),
        "state_error_after_execute": float(state_error_after_execute),
    }
