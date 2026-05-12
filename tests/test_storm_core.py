import numpy as np

from storm.core import StructuredTrajectory, TrajectoryComponent
from storm.planners import StraightLinePlanner
from storm.core import PlanningContext


def test_structured_trajectory_joint_array() -> None:
    base = TrajectoryComponent("base", np.zeros((4, 3)), ("x", "y", "yaw"))
    arm = TrajectoryComponent("arm", np.ones((4, 2)), ("q1", "q2"))
    trajectory = StructuredTrajectory({"base": base, "arm": arm})

    assert trajectory.horizon == 4
    assert trajectory.as_joint_array(("base", "arm")).shape == (4, 5)


def test_straight_line_planner_contract() -> None:
    planner = StraightLinePlanner()
    context = PlanningContext(start=np.array([0.0, 0.0]), goals=np.array([1.0, 1.0]))
    result = planner.plan(context, {"horizon": 8})

    assert result.success
    assert result.trajectory is not None
    assert result.trajectory.horizon == 8
    np.testing.assert_allclose(result.trajectory.component("robot").values[-1], np.array([1.0, 1.0]))
