from __future__ import annotations

PLANNED_BASELINES = (
    "rrt_connect",
    "rrt_connect_receding",
    "shift_previous_plan",
    "shift_previous_plan_plus_smoothing",
    "trajectory_optimization",
    "diffusion_noise_to_traj",
    "diffusion_plan_to_plan",
    "flow_matching_plan_to_plan",
)


def is_registered_baseline(name: str) -> bool:
    return name in PLANNED_BASELINES
