from pathlib import Path

from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D
from mm_flow.three_d.metrics import compute_plan_metrics
from mm_flow.three_d.planners import RecedingRRTConnectConfig, available_planners, plan_problem
from mm_flow.three_d.problems import build_moma_reach_sequence_problem
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.run_io import save_reach_sequence_run


def test_moma_problem_and_rrt_pipeline_smoke(tmp_path: Path) -> None:
    robot = MomaPiperMobileManipulator3D()
    problem = build_moma_reach_sequence_problem(robot=robot, seed=3, obstacle_count=2)
    assert problem.name == "moma_reach_sequence"
    assert problem.goal_count == 3
    assert problem.obstacle_count == 2
    assert "rrt_connect" in available_planners()

    result = plan_problem(
        "rrt_connect",
        robot=robot,
        problem=problem,
        config=RRTConnectConfig(
            max_iterations=2,
            goal_sample_count=2,
            shortcut_attempts=0,
            rng_seed=3,
            clearance_margin=0.01,
            bounds_xy=problem.bounds_xy,
        ),
    )
    assert result.planner_name == "rrt_connect"
    assert len(result.segment_ends) == problem.goal_count
    assert result.trajectory_steps >= 1
    assert len(result.planning_times) == problem.goal_count
    assert result.metadata["collision_check_count"] >= 1
    assert len(result.metadata["per_goal_collision_checks"]) == problem.goal_count

    metrics = compute_plan_metrics(problem, result)
    assert len(metrics.per_goal_path_length) == problem.goal_count
    assert len(metrics.per_goal_base_path_length) == problem.goal_count
    assert len(metrics.per_goal_joint_motion_length) == problem.goal_count
    assert len(metrics.per_goal_end_effector_path_length) == problem.goal_count
    assert len(metrics.per_goal_min_clearance) == problem.goal_count

    paths = save_reach_sequence_run(robot, problem, result, tmp_path, "RRT-Connect")
    assert paths["config"].exists()
    assert paths["scene"].exists()
    assert paths["result"].exists()
    assert paths["trajectory"].exists()
    assert paths["html"].exists()


def test_receding_rrt_planner_smoke() -> None:
    robot = MomaPiperMobileManipulator3D()
    problem = build_moma_reach_sequence_problem(robot=robot, seed=3, obstacle_count=2)
    assert "rrt_connect_receding" in available_planners()

    result = plan_problem(
        "rrt_connect_receding",
        robot=robot,
        problem=problem,
        config=RecedingRRTConnectConfig(
            rrt=RRTConnectConfig(
                max_iterations=2,
                goal_sample_count=2,
                shortcut_attempts=0,
                rng_seed=3,
                clearance_margin=0.01,
                bounds_xy=problem.bounds_xy,
            ),
            horizon_steps=4,
            execute_steps=2,
            max_cycles_per_goal=2,
        ),
    )
    assert result.planner_name == "rrt_connect_receding"
    assert len(result.segment_ends) == problem.goal_count
    assert result.trajectory_steps >= 1
    assert "replanning_count" in result.metadata
    assert "receding_cycles" in result.metadata
    assert len(result.metadata["receding_cycles"]) >= 1
    assert "frame_start" in result.metadata["receding_cycles"][0]
    assert "state_error_after_execute" in result.metadata["receding_cycles"][0]


def test_moma_problem_variants_smoke() -> None:
    robot = MomaPiperMobileManipulator3D()
    for variant in ["standard", "narrow_passage", "arm_obstacle", "base_required"]:
        problem = build_moma_reach_sequence_problem(robot=robot, seed=2, obstacle_count=6, variant=variant)
        assert problem.metadata["variant"] == variant
        assert problem.goal_count == 3
        assert problem.obstacle_count == 6
