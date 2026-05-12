from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class TrajectoryComponent:
    """One named part of a structured trajectory bundle."""

    name: str
    values: np.ndarray
    state_fields: tuple[str, ...] = ()
    frame: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"component {self.name!r} must be a 2D array, got shape {values.shape}")
        if self.state_fields and len(self.state_fields) != values.shape[1]:
            raise ValueError(
                f"component {self.name!r} has {values.shape[1]} state dimensions, "
                f"but {len(self.state_fields)} state fields were provided"
            )
        object.__setattr__(self, "values", values)

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class StructuredTrajectory:
    """A joint trajectory object whose components are generated together."""

    components: Mapping[str, TrajectoryComponent]
    dt: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        components = dict(self.components)
        if not components:
            raise ValueError("structured trajectory must contain at least one component")
        horizons = {component.horizon for component in components.values()}
        if len(horizons) != 1:
            raise ValueError(f"all components must share the same horizon, got {sorted(horizons)}")
        for name, component in components.items():
            if name != component.name:
                raise ValueError(f"component mapping key {name!r} does not match component name {component.name!r}")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be positive when provided")
        object.__setattr__(self, "components", components)

    @property
    def horizon(self) -> int:
        return next(iter(self.components.values())).horizon

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.components.keys())

    def component(self, name: str) -> TrajectoryComponent:
        return self.components[name]

    def as_joint_array(self, order: tuple[str, ...] | None = None) -> np.ndarray:
        names = order or self.names
        return np.concatenate([self.components[name].values for name in names], axis=1)


@dataclass(frozen=True)
class PlanningContext:
    """Conditioning information shared by generative and classical planners."""

    start: np.ndarray
    goals: np.ndarray
    environment: dict[str, Any] = field(default_factory=dict)
    history: StructuredTrajectory | None = None
    robot: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", np.asarray(self.start, dtype=float))
        goals = np.asarray(self.goals, dtype=float)
        if goals.ndim == 1:
            goals = goals[None, :]
        if goals.ndim != 2:
            raise ValueError(f"goals must be a 1D or 2D array, got shape {goals.shape}")
        object.__setattr__(self, "goals", goals)


@dataclass(frozen=True)
class ConstraintReport:
    name: str
    value: float
    feasible: bool
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanResult:
    planner_name: str
    success: bool
    trajectory: StructuredTrajectory | None
    planning_time_s: float
    constraints: tuple[ConstraintReport, ...] = ()
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def constraint_feasible(self) -> bool:
        return all(report.feasible for report in self.constraints)


class GenerativeTrajectoryPlanner(Protocol):
    """Minimal planner API for diffusion, flow matching, and classical baselines."""

    name: str

    def plan(self, context: PlanningContext, config: Mapping[str, Any] | None = None) -> PlanResult:
        ...
