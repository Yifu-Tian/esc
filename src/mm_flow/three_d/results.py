from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReachSequencePlanResult:
    trajectory: np.ndarray
    segment_ends: list[int]
    terminal_errors: list[float]
    clearances: list[float]
    iterations: list[int]
    planning_times: list[float]
    messages: list[str]
    success: bool
    planner_name: str
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_planning_time(self) -> float:
        return float(sum(self.planning_times))

    @property
    def trajectory_steps(self) -> int:
        return int(len(self.trajectory))

    @property
    def planning_time(self) -> float:
        return self.total_planning_time

    @property
    def per_goal_errors(self) -> list[float]:
        return self.terminal_errors

    @property
    def min_clearance(self) -> float:
        if not self.clearances:
            return float("nan")
        return float(min(self.clearances))

    @property
    def mean_clearance(self) -> float:
        if not self.clearances:
            return float("nan")
        return float(np.mean(self.clearances))

    @property
    def iteration_count(self) -> int:
        return int(sum(self.iterations))
