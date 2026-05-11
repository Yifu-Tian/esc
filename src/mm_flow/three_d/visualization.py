from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, Obstacle3D, SphereObstacle
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D


def plot_reach_sequence_3d(
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    goals_xyz: np.ndarray,
    segment_ends: list[int],
    obstacles: list[Obstacle3D],
    output_path: str | Path,
    title: str = "3D Mobile Manipulator Reach Sequence",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, robot, trajectory, goals_xyz, obstacles, title)
    _draw_obstacles(ax, obstacles)
    _draw_goals(ax, goals_xyz)

    base_path = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))])
    ee_path = np.array([robot.end_effector(state) for state in trajectory])
    ax.plot(base_path[:, 0], base_path[:, 1], base_path[:, 2], color="#4c72b0", linewidth=2.0, label="base path")
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="#55a868", linewidth=2.0, label="end-effector path")

    frame_ids = sorted(set(np.linspace(0, len(trajectory) - 1, 12, dtype=int).tolist() + segment_ends))
    colors = ["#222222", "#8172b3", "#ccb974", "#64b5cd"]
    for idx in frame_ids:
        fk = robot.forward_kinematics(trajectory[idx])
        alpha = 0.2 + 0.65 * idx / max(len(trajectory) - 1, 1)
        color = colors[min(sum(idx > end for end in segment_ends), len(colors) - 1)]
        _draw_robot(ax, robot, fk, color=color, alpha=alpha)

    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def animate_reach_sequence_3d(
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    goals_xyz: np.ndarray,
    segment_ends: list[int],
    obstacles: list[Obstacle3D],
    output_path: str | Path,
    title: str = "3D Mobile Manipulator Reach Sequence",
    fps: int = 12,
    stride: int = 1,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, robot, trajectory, goals_xyz, obstacles, title)
    _draw_obstacles(ax, obstacles)
    _draw_goals(ax, goals_xyz)

    base_path = np.column_stack([trajectory[:, 0], trajectory[:, 1], np.zeros(len(trajectory))])
    ee_path = np.array([robot.end_effector(state) for state in trajectory])
    base_trace, = ax.plot([], [], [], color="#4c72b0", linewidth=2.0, label="base path")
    ee_trace, = ax.plot([], [], [], color="#55a868", linewidth=2.0, label="end-effector path")
    arm_line, = ax.plot([], [], [], color="#222222", linewidth=3.0, marker="o", markersize=4)
    base_point, = ax.plot([], [], [], marker="o", color="#4c72b0", markersize=9)
    ee_point, = ax.plot([], [], [], marker="o", color="#55a868", markersize=6)
    info_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    frame_ids = list(range(0, len(trajectory), max(stride, 1)))
    if frame_ids[-1] != len(trajectory) - 1:
        frame_ids.append(len(trajectory) - 1)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=110):
        for idx in frame_ids:
            fk = robot.forward_kinematics(trajectory[idx])
            mount = fk["mount"]
            elbow = fk["elbow"]
            ee = fk["ee"]
            base = fk["base"]
            segment_idx = min(sum(idx > end for end in segment_ends), len(goals_xyz) - 1)

            base_trace.set_data(base_path[: idx + 1, 0], base_path[: idx + 1, 1])
            base_trace.set_3d_properties(base_path[: idx + 1, 2])
            ee_trace.set_data(ee_path[: idx + 1, 0], ee_path[: idx + 1, 1])
            ee_trace.set_3d_properties(ee_path[: idx + 1, 2])
            arm_line.set_data([mount[0], elbow[0], ee[0]], [mount[1], elbow[1], ee[1]])
            arm_line.set_3d_properties([mount[2], elbow[2], ee[2]])
            base_point.set_data([base[0]], [base[1]])
            base_point.set_3d_properties([base[2]])
            ee_point.set_data([ee[0]], [ee[1]])
            ee_point.set_3d_properties([ee[2]])
            info_text.set_text(f"step {idx}/{len(trajectory)-1} | target g{segment_idx + 1}")
            writer.grab_frame()

    plt.close(fig)


def _draw_robot(ax, robot: SimpleMobileManipulator3D, fk: dict[str, np.ndarray], color: str, alpha: float) -> None:
    base = fk["base"]
    mount = fk["mount"]
    elbow = fk["elbow"]
    ee = fk["ee"]
    ax.scatter([base[0]], [base[1]], [base[2]], color="#4c72b0", s=32, alpha=alpha)
    ax.plot([mount[0], elbow[0], ee[0]], [mount[1], elbow[1], ee[1]], [mount[2], elbow[2], ee[2]],
            color=color, alpha=alpha, linewidth=2.4, marker="o", markersize=3)


def _draw_goals(ax, goals_xyz: np.ndarray) -> None:
    for i, goal in enumerate(goals_xyz):
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="#55a868", marker="*", s=120)
        ax.text(goal[0], goal[1], goal[2] + 0.06, f"g{i + 1}", color="#2f7d43", fontsize=9)


def _draw_obstacles(ax, obstacles: list[Obstacle3D]) -> None:
    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            _draw_sphere(ax, np.array(obs.center, dtype=float), obs.radius)
        elif isinstance(obs, CuboidObstacle):
            _draw_cuboid(ax, np.array(obs.center, dtype=float), np.array(obs.size, dtype=float))
        elif isinstance(obs, CylinderObstacle):
            _draw_cylinder(ax, np.array(obs.center, dtype=float), obs.radius, obs.height)


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
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
             (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="#8b1a1a", alpha=0.45, linewidth=1.0)


def _draw_cylinder(ax, center: np.ndarray, radius: float, height: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 28)
    z = np.array([center[2] - 0.5 * height, center[2] + 0.5 * height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color="#c44e52", alpha=0.22, linewidth=0)
    for zc in z:
        ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
                np.full_like(theta, zc), color="#8b1a1a", alpha=0.45, linewidth=0.8)


def _setup_axes(
    ax,
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    goals_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
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
        elif isinstance(obs, CylinderObstacle):
            center = np.array(obs.center, dtype=float)
            half = np.array([obs.radius, obs.radius, 0.5 * obs.height], dtype=float)
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
