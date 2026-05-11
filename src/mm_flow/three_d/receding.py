from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.results import ReachSequencePlanResult


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
    def __init__(self, horizon_steps: int, execute_steps: int) -> None:
        self.horizon_steps = horizon_steps
        self.execute_steps = execute_steps

    def run(self, problem: ReachSequenceProblem, planner_name: str, config: object) -> ReachSequencePlanResult:
        raise NotImplementedError("Receding horizon planning is not implemented yet.")
