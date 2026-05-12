from __future__ import annotations

import numpy as np

from storm.topology import CircularObstacle2D, topology_reports


def main() -> None:
    history = np.array([[0.0, 0.0], [1.0, -0.8], [2.0, 0.0]])
    candidate = np.array([[2.0, 0.0], [2.0, 1.0], [1.0, 1.2], [0.0, 1.0]])
    obstacles = [CircularObstacle2D(center=np.array([1.0, 0.2]), radius=0.25, name="o1")]
    for report in topology_reports(history, candidate, obstacles):
        print(f"{report.name}: value={report.value:.3f}, feasible={report.feasible}")


if __name__ == "__main__":
    main()
