import numpy as np

from esc.history_graph import CircleObstacle, build_history_graph, split_polyline


def test_split_polyline_returns_requested_segment_count() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    segments = split_polyline(path, 4)

    assert segments.shape == (4, 2, 2)
    np.testing.assert_allclose(segments[0, 0], path[0])
    np.testing.assert_allclose(segments[-1, -1], path[-1])


def test_build_history_graph_shapes() -> None:
    theta = np.linspace(0.0, np.pi, 50)
    history = np.column_stack([np.cos(theta), np.sin(theta)])
    obstacles = [
        CircleObstacle(name="o1", center=np.array([0.0, 0.0]), radius=0.2),
        CircleObstacle(name="o2", center=np.array([2.0, 0.0]), radius=0.2),
    ]

    graph = build_history_graph(history, obstacles, segment_count=8)

    assert graph["segments"].shape == (8, 2, 2)
    assert graph["tokens"].shape == (8, 12)
    assert graph["history_obstacle"]["clearance"].shape == (8, 2)
    assert graph["history_history"]["distance"].shape == (8, 8)
