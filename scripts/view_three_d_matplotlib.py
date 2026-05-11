#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib


def _select_gui_backend() -> None:
    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        return
    candidates = [
        ("Qt5Agg", "PyQt5"),
        ("TkAgg", "tkinter"),
    ]
    for backend, module in candidates:
        if importlib.util.find_spec(module) is None:
            continue
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


_select_gui_backend()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mm_flow.three_d.collision import CuboidObstacle, SphereObstacle
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D
from mm_flow.three_d.tasks import build_reach_sequence_scene_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Matplotlib 3D viewer for a saved 3D reach sequence.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("outputs/three_d_reach_sequence/reach_sequence_3d.npz"),
    )
    parser.add_argument("--frame", type=int, default=-1, help="Frame to display. Use -1 for final frame.")
    args = parser.parse_args()

    if "agg" in matplotlib.get_backend().lower():
        raise RuntimeError(
            "Matplotlib is using a non-GUI backend. Run this from a graphical desktop session, "
            "or install a GUI backend such as tkinter/Qt. The HTML viewer remains available at "
            "outputs/three_d_reach_sequence/reach_sequence_3d.html."
        )

    robot = SimpleMobileManipulator3D()
    _, _, obstacles = build_reach_sequence_scene_3d()
    data = np.load(args.trajectory)
    trajectory = data["trajectory"]
    goals_xyz = data["goals_xyz"]
    frame = args.frame if args.frame >= 0 else len(trajectory) - 1

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, robot, trajectory, goals_xyz, obstacles, "Interactive Matplotlib 3D Viewer")
    _draw_obstacles(ax, obstacles)
    _draw_goals(ax, goals_xyz)

    base_path = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))])
    ee_path = np.array([robot.end_effector(state) for state in trajectory])
    ax.plot(base_path[:, 0], base_path[:, 1], base_path[:, 2], color="#4c72b0", linewidth=2.0, label="base path")
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="#55a868", linewidth=2.0, label="end-effector path")
    _draw_robot(ax, robot.forward_kinematics(trajectory[frame]), color="#222222", alpha=0.9)
    ax.legend(loc="upper left", fontsize=8)
    plt.show()


def _draw_robot(ax, fk: dict[str, np.ndarray], color: str, alpha: float) -> None:
    base = fk["base"]
    mount = fk["mount"]
    elbow = fk["elbow"]
    ee = fk["ee"]
    ax.scatter([base[0]], [base[1]], [base[2]], color="#4c72b0", s=32, alpha=alpha)
    ax.plot(
        [mount[0], elbow[0], ee[0]],
        [mount[1], elbow[1], ee[1]],
        [mount[2], elbow[2], ee[2]],
        color=color,
        alpha=alpha,
        linewidth=2.4,
        marker="o",
        markersize=3,
    )


def _draw_goals(ax, goals_xyz: np.ndarray) -> None:
    for i, goal in enumerate(goals_xyz):
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="#55a868", marker="*", s=120)
        ax.text(goal[0], goal[1], goal[2] + 0.06, f"g{i + 1}", color="#2f7d43", fontsize=9)


def _draw_obstacles(ax, obstacles: list[SphereObstacle | CuboidObstacle]) -> None:
    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            _draw_sphere(ax, np.array(obs.center, dtype=float), obs.radius)
        elif isinstance(obs, CuboidObstacle):
            _draw_cuboid(ax, np.array(obs.center, dtype=float), np.array(obs.size, dtype=float))


def _draw_sphere(ax, center: np.ndarray, radius: float) -> None:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#c44e52", alpha=0.22, linewidth=0)


def _draw_cuboid(ax, center: np.ndarray, size: np.ndarray) -> None:
    half = 0.5 * size
    x = [center[0] - half[0], center[0] + half[0]]
    y = [center[1] - half[1], center[1] + half[1]]
    z = [center[2] - half[2], center[2] + half[2]]
    corners = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])
    edges = [
        (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
    ]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="#8b1a1a", alpha=0.45, linewidth=1.0)


def _setup_axes(
    ax,
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    goals_xyz: np.ndarray,
    obstacles: list[SphereObstacle | CuboidObstacle],
    title: str,
) -> None:
    points = [np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))]), goals_xyz]
    for state in trajectory:
        fk = robot.forward_kinematics(state)
        points.extend([fk["mount"][None, :], fk["elbow"][None, :], fk["ee"][None, :]])
    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            center = np.array(obs.center, dtype=float)
            r = obs.radius
            points.append(np.array([center - r, center + r]))
        elif isinstance(obs, CuboidObstacle):
            center = np.array(obs.center, dtype=float)
            half = 0.5 * np.array(obs.size, dtype=float)
            points.append(np.array([center - half, center + half]))

    all_points = np.vstack(points)
    min_xyz = all_points.min(axis=0) - 0.45
    max_xyz = all_points.max(axis=0) + 0.45
    span = max_xyz - min_xyz
    max_span = float(np.max(span))
    center = 0.5 * (min_xyz + max_xyz)
    half = 0.5 * max_span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.0, center[2] - half), center[2] + half)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)


if __name__ == "__main__":
    main()

