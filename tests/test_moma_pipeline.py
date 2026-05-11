from pathlib import Path

from mm_flow.three_d.moma_kinematics import MomaPiperMobileManipulator3D
from mm_flow.three_d.planners import available_planners, plan_problem
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

    paths = save_reach_sequence_run(robot, problem, result, tmp_path, "RRT-Connect")
    assert paths["config"].exists()
    assert paths["scene"].exists()
    assert paths["result"].exists()
    assert paths["trajectory"].exists()
    assert paths["html"].exists()
