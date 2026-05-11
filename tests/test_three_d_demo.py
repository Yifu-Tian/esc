from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, SphereObstacle
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.rrt_connect import RRTConnectConfig
from mm_flow.three_d.tasks import build_reach_sequence_scene_3d, plan_reach_sequence_3d


def test_random_three_d_scene_contains_supported_obstacle_types():
    _, _, obstacles = build_reach_sequence_scene_3d(seed=7, random_obstacle_count=16)

    assert len(obstacles) == 16
    assert any(isinstance(obs, SphereObstacle) for obs in obstacles)
    assert any(isinstance(obs, CuboidObstacle) for obs in obstacles)
    assert any(isinstance(obs, CylinderObstacle) for obs in obstacles)
    assert any(_is_floating(obs) for obs in obstacles)
    assert any(_is_grounded(obs) for obs in obstacles)


def test_three_d_reach_sequence_generates_complete_trajectory():
    robot = SimpleMobileManipulator3D()
    start, goals_xyz, obstacles = build_reach_sequence_scene_3d()

    trajectory, segment_ends, terminal_errors, clearances, success = plan_reach_sequence_3d(
        robot=robot,
        start=start,
        goals_xyz=goals_xyz,
        obstacles=obstacles,
        segment_horizon=40,
    )

    assert len(segment_ends) == len(goals_xyz)
    assert trajectory.shape[1] == robot.state_dim
    assert max(terminal_errors) < 0.09
    assert all(end > 0 for end in segment_ends)
    assert isinstance(success, bool)


def test_rrt_connect_reach_sequence_finds_collision_free_path_in_sparse_scene():
    robot = SimpleMobileManipulator3D()
    start, goals_xyz, obstacles = build_reach_sequence_scene_3d(seed=2, random_obstacle_count=0)

    trajectory, segment_ends, terminal_errors, clearances, success = plan_reach_sequence_3d(
        robot=robot,
        start=start,
        goals_xyz=goals_xyz[:1],
        obstacles=obstacles,
        segment_horizon=40,
        planner="rrt_connect",
        rrt_config=RRTConnectConfig(max_iterations=1200, goal_sample_count=40, shortcut_attempts=10, rng_seed=2),
    )

    assert success
    assert len(segment_ends) == 1
    assert max(terminal_errors) < 0.09
    assert min(clearances) >= 0.0
    assert trajectory.shape[1] == robot.state_dim


def _is_grounded(obstacle) -> bool:
    if isinstance(obstacle, SphereObstacle):
        return abs(obstacle.center[2] - obstacle.radius) < 1e-6
    if isinstance(obstacle, CuboidObstacle):
        return abs(obstacle.center[2] - 0.5 * obstacle.size[2]) < 1e-6
    if isinstance(obstacle, CylinderObstacle):
        return abs(obstacle.center[2] - 0.5 * obstacle.height) < 1e-6
    return False


def _is_floating(obstacle) -> bool:
    return not _is_grounded(obstacle)
