import numpy as np

from storm.topology import CircularObstacle2D, concatenate_history_candidate, fractional_winding_number, topology_reports


def test_fractional_winding_number_detects_one_loop() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    assert fractional_winding_number(circle, np.array([0.0, 0.0])) > 0.99


def test_history_candidate_concatenation_drops_duplicate_join() -> None:
    history = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidate = np.array([[1.0, 0.0], [2.0, 0.0]])
    merged = concatenate_history_candidate(history, candidate)

    assert merged.shape == (3, 2)


def test_topology_reports_mark_large_winding_infeasible() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    candidate = np.column_stack([np.cos(theta), np.sin(theta)])
    reports = topology_reports(
        history_xy=None,
        candidate_xy=candidate,
        obstacles=[CircularObstacle2D(center=np.array([0.0, 0.0]), radius=0.2, name="center")],
        winding_threshold=0.95,
    )

    max_report = [report for report in reports if report.name == "max_abs_winding"][0]
    assert not max_report.feasible
