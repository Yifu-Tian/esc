from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import numpy as np

from mm_flow.three_d.collision import Obstacle3D
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.receding import RecedingHorizonRunner
from mm_flow.three_d.results import ReachSequencePlanResult
from mm_flow.three_d.rrt_connect import RRTConnectConfig, plan_rrt_connect_to_goal


@dataclass(frozen=True)
class RecedingRRTConnectConfig:
    rrt: RRTConnectConfig
    horizon_steps: int = 24
    execute_steps: int = 6
    max_cycles_per_goal: int = 8
    follower_alpha: float = 1.0
    disturbance_std: float = 0.0
    follower_seed: int = 0


@dataclass(frozen=True)
class PlannerSpec:
    name: str
    display_name: str
    config_type: type
    planner: "Planner"


class Planner(ABC):
    name: str
    display_name: str
    config_type: type

    @abstractmethod
    def plan(self, problem: ReachSequenceProblem, config: object) -> ReachSequencePlanResult:
        raise NotImplementedError


class RRTConnectPlanner(Planner):
    name = "rrt_connect"
    display_name = "RRT-Connect whole-body baseline"
    config_type = RRTConnectConfig

    def plan(self, problem: ReachSequenceProblem, config: object) -> ReachSequencePlanResult:
        if not isinstance(config, RRTConnectConfig):
            raise TypeError(f"{self.name} expects config type RRTConnectConfig")
        return plan_rrt_connect_reach_sequence(problem, config)


class RecedingRRTConnectPlanner(Planner):
    name = "rrt_connect_receding"
    display_name = "Receding-horizon RRT-Connect baseline"
    config_type = RecedingRRTConnectConfig

    def plan(self, problem: ReachSequenceProblem, config: object) -> ReachSequencePlanResult:
        if not isinstance(config, RecedingRRTConnectConfig):
            raise TypeError(f"{self.name} expects config type RecedingRRTConnectConfig")
        from mm_flow.three_d.execution import FirstOrderTrajectoryFollower

        runner = RecedingHorizonRunner(
            horizon_steps=config.horizon_steps,
            execute_steps=config.execute_steps,
            max_cycles_per_goal=config.max_cycles_per_goal,
            follower=FirstOrderTrajectoryFollower(
                alpha=config.follower_alpha,
                disturbance_std=config.disturbance_std,
                seed=config.follower_seed,
            ),
        )
        return runner.run(problem, "rrt_connect", config.rrt)


def available_planners() -> tuple[str, ...]:
    return tuple(PLANNER_REGISTRY.keys())


def get_planner(name: str) -> PlannerSpec:
    try:
        return PLANNER_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(available_planners())
        raise ValueError(f"Unknown planner '{name}'. Available planners: {valid}") from exc


def plan_reach_sequence(
    planner_name: str,
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goals_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    config: object,
) -> ReachSequencePlanResult:
    problem = ReachSequenceProblem(
        name="adhoc_reach_sequence",
        robot=robot,
        start=start,
        goals_xyz=goals_xyz,
        obstacles=obstacles,
        seed=-1,
        bounds_xy=getattr(config, "bounds_xy", ((-2.8, 2.8), (-1.9, 1.9))),
    )
    return plan_problem(planner_name, robot, problem, config)


def plan_problem(
    planner_name: str,
    robot: SimpleMobileManipulator3D,
    problem: ReachSequenceProblem,
    config: object,
) -> ReachSequencePlanResult:
    planner = get_planner(planner_name)
    if not isinstance(config, planner.config_type):
        raise TypeError(f"Planner '{planner_name}' expects config type {planner.config_type.__name__}")
    return planner.planner.plan(problem, config)


def plan_rrt_connect_reach_sequence(
    problem: ReachSequenceProblem,
    config: RRTConnectConfig,
) -> ReachSequencePlanResult:
    robot = problem.robot
    start = problem.start
    goals_xyz = problem.goals_xyz
    obstacles = problem.obstacles
    current = start.copy()
    pieces = []
    segment_ends: list[int] = []
    terminal_errors: list[float] = []
    clearances: list[float] = []
    iterations: list[int] = []
    planning_times: list[float] = []
    messages: list[str] = []
    per_goal_collision_checks: list[int] = []
    success = True

    for i, goal_xyz in enumerate(goals_xyz):
        segment_config = RRTConnectConfig(
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
        start_time = time.perf_counter()
        result = plan_rrt_connect_to_goal(robot, current, goal_xyz, obstacles, segment_config)
        planning_times.append(time.perf_counter() - start_time)

        segment = result.trajectory if i == 0 else result.trajectory[1:]
        pieces.append(segment)
        current = result.trajectory[-1].copy()
        terminal_errors.append(result.terminal_error)
        clearances.append(result.min_clearance)
        iterations.append(result.iterations)
        messages.append(result.message)
        per_goal_collision_checks.append(result.collision_checks)
        segment_ends.append(sum(len(piece) for piece in pieces) - 1)
        success = success and result.success
        if not result.success:
            break

        if i < len(goals_xyz) - 1:
            dwell = np.repeat(current[None, :], 4, axis=0)
            pieces.append(dwell)

    while len(segment_ends) < len(goals_xyz):
        goal_xyz = goals_xyz[len(segment_ends)]
        terminal_errors.append(float(np.linalg.norm(robot.end_effector(current) - goal_xyz)))
        clearances.append(float(result.min_clearance if clearances else 0.0))
        iterations.append(0)
        planning_times.append(0.0)
        messages.append("skipped after failed segment")
        per_goal_collision_checks.append(0)
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
        planner_name="rrt_connect",
        failure_reason="" if success else "; ".join(dict.fromkeys(message for message in messages if message != "connected")),
        metadata={
            "collision_checker": "proxy",
            "task_type": problem.task_type,
            "collision_check_count": int(sum(per_goal_collision_checks)),
            "per_goal_collision_checks": per_goal_collision_checks,
        },
    )


_RRT_CONNECT_PLANNER = RRTConnectPlanner()
_RECEDING_RRT_CONNECT_PLANNER = RecedingRRTConnectPlanner()

PLANNER_REGISTRY: dict[str, PlannerSpec] = {
    "rrt_connect": PlannerSpec(
        name=_RRT_CONNECT_PLANNER.name,
        display_name=_RRT_CONNECT_PLANNER.display_name,
        config_type=RRTConnectConfig,
        planner=_RRT_CONNECT_PLANNER,
    ),
    "rrt_connect_receding": PlannerSpec(
        name=_RECEDING_RRT_CONNECT_PLANNER.name,
        display_name=_RECEDING_RRT_CONNECT_PLANNER.display_name,
        config_type=RecedingRRTConnectConfig,
        planner=_RECEDING_RRT_CONNECT_PLANNER,
    ),
}
