from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionResult:
    executed_states: np.ndarray
    tracking_errors: np.ndarray


class TrajectoryFollower:
    def execute(self, trajectory: np.ndarray, initial_state: np.ndarray) -> ExecutionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class FirstOrderTrajectoryFollower(TrajectoryFollower):
    alpha: float = 0.75
    disturbance_std: float = 0.0
    seed: int = 0

    def execute(self, trajectory: np.ndarray, initial_state: np.ndarray) -> ExecutionResult:
        rng = np.random.default_rng(self.seed)
        current = np.asarray(initial_state, dtype=float).copy()
        executed = []
        errors = []
        for target in np.asarray(trajectory, dtype=float):
            noise = rng.normal(0.0, self.disturbance_std, size=current.shape)
            current = current + self.alpha * (target - current) + noise
            executed.append(current.copy())
            errors.append(float(np.linalg.norm(current - target)))
        if not executed:
            return ExecutionResult(executed_states=current[None, :], tracking_errors=np.zeros(1, dtype=float))
        return ExecutionResult(executed_states=np.vstack(executed), tracking_errors=np.array(errors, dtype=float))
