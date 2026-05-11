from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mm_flow.three_d.problems import ReachSequenceProblem
from mm_flow.three_d.results import ReachSequencePlanResult
from mm_flow.three_d.serialization import obstacle_to_dict, to_jsonable


@dataclass(frozen=True)
class ExpertExample:
    condition: dict
    target: dict


def build_expert_example(
    problem: ReachSequenceProblem,
    result: ReachSequencePlanResult,
    previous_plan: np.ndarray | None = None,
    observation: dict | None = None,
) -> ExpertExample:
    condition = {
        "task_type": problem.task_type,
        "start": problem.start,
        "goals": problem.goals_xyz,
        "obstacles": [obstacle_to_dict(obstacle) for obstacle in problem.obstacles],
        "previous_plan": previous_plan,
        "observation": observation or {},
    }
    target = {
        "whole_body_trajectory": result.trajectory,
        "segment_ends": result.segment_ends,
        "success": result.success,
    }
    return ExpertExample(condition=to_jsonable(condition), target=to_jsonable(target))


def save_expert_example(path: Path, example: ExpertExample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, condition=example.condition, target=example.target)
