from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from storm.constraints import goal_reached_report
from storm.core import PlanningContext, PlanResult, StructuredTrajectory, TrajectoryComponent


@dataclass(frozen=True)
class StraightLineConfig:
    horizon: int = 32
    goal_tolerance: float = 1e-6


class StraightLinePlanner:
    """Small baseline used to exercise the STORM planner contract."""

    name = "straight_line"

    def plan(self, context: PlanningContext, config: Mapping[str, Any] | None = None) -> PlanResult:
        start_time = perf_counter()
        cfg = StraightLineConfig(**dict(config or {}))
        if cfg.horizon < 2:
            raise ValueError("horizon must be at least 2")

        start = np.asarray(context.start, dtype=float)
        goal = np.asarray(context.goals[0], dtype=float)
        values = np.linspace(start, goal, cfg.horizon)
        trajectory = StructuredTrajectory(
            components={
                "robot": TrajectoryComponent(
                    name="robot",
                    values=values,
                    state_fields=tuple(f"x{i}" for i in range(values.shape[1])),
                )
            },
            metadata={"representation": "single_point_robot"},
        )
        goal_report = goal_reached_report(values[-1], goal, cfg.goal_tolerance)
        return PlanResult(
            planner_name=self.name,
            success=goal_report.feasible,
            trajectory=trajectory,
            planning_time_s=perf_counter() - start_time,
            constraints=(goal_report,),
        )
