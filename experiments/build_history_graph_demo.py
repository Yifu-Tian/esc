from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from esc.history_graph import (
    CircleObstacle,
    arrays_to_lists,
    build_history_graph,
    fractional_winding_number,
    polyline_length,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and visualize an ESC history topological memory graph.")
    parser.add_argument("--case", type=Path, default=Path("/home/yifu/storm/outputs/manual_2d_obstacle_scene/manual_2d_tangle_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/history_graph_demo"))
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--robot-radius", type=float, default=0.03)
    parser.add_argument("--near-threshold", type=float, default=0.08)
    args = parser.parse_args()

    case = json.loads(args.case.read_text(encoding="utf-8"))
    scene = case.get("source_scene", case)
    bounds = tuple(float(value) for value in scene["bounds"])
    obstacles = [CircleObstacle.from_dict(item) for item in scene["obstacles"]]
    history = np.asarray(case["history"], dtype=float)
    graph = build_history_graph(
        history,
        obstacles,
        segment_count=args.segments,
        robot_radius=args.robot_radius,
        near_threshold=args.near_threshold,
    )

    windings = {obstacle.name: fractional_winding_number(history, obstacle.center) for obstacle in obstacles}
    nearest_obstacle_indices = graph["tokens"][:, -1].astype(int)
    nearest_counts = {
        obstacles[index].name: int(np.sum(nearest_obstacle_indices == index))
        for index in range(len(obstacles))
    }
    summary = {
        "case_path": str(args.case),
        "history_points": int(len(history)),
        "history_length": polyline_length(history),
        "segment_count": int(graph["segment_count"]),
        "token_schema": [
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "mid_x",
            "mid_y",
            "dir_x",
            "dir_y",
            "length",
            "normalized_time",
            "min_obstacle_clearance",
            "nearest_obstacle_index",
        ],
        "history_obstacle_relation_schema": [
            "distance",
            "clearance",
            "signed_side",
            "local_angle",
            "near",
            "collision",
        ],
        "history_history_relation_schema": [
            "distance",
            "crossing",
            "near",
            "antiparallel",
            "temporal_gap",
        ],
        "min_history_obstacle_clearance": float(np.min(graph["history_obstacle"]["clearance"])),
        "near_history_obstacle_edges": int(np.sum(graph["history_obstacle"]["near"])),
        "history_history_near_edges": int(np.sum(graph["history_history"]["near"])),
        "history_history_crossing_edges": int(np.sum(graph["history_history"]["crossing"])),
        "obstacle_winding": windings,
        "nearest_obstacle_segment_counts": nearest_counts,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_payload = {
        "summary": summary,
        "obstacles": [
            {"name": obstacle.name, "center": obstacle.center.tolist(), "radius": obstacle.radius}
            for obstacle in obstacles
        ],
        "history_graph": arrays_to_lists(graph),
    }
    (args.output_dir / "history_graph.json").write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_history_graph_scene(bounds, obstacles, graph, history, args.robot_radius, args.output_dir / "history_graph_scene.png")
    render_relation_heatmaps(graph, obstacles, args.output_dir / "history_relation_heatmaps.png")
    print(json.dumps(summary, indent=2))


def render_history_graph_scene(
    bounds: tuple[float, float, float, float],
    obstacles: list[CircleObstacle],
    graph: dict,
    history: np.ndarray,
    robot_radius: float,
    output_path: Path,
) -> None:
    segments = graph["segments"]
    tokens = graph["tokens"]
    near_ho = graph["history_obstacle"]["near"]
    crossing_hh = graph["history_history"]["crossing"]
    near_hh = graph["history_history"]["near"]

    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    xmin, xmax, ymin, ymax = bounds
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1.4, edgecolor="#475569"))
    for obstacle in obstacles:
        ax.add_patch(Circle(obstacle.center, obstacle.radius + robot_radius, color="#ef4444", alpha=0.12, linewidth=0))
        ax.add_patch(Circle(obstacle.center, obstacle.radius, fill=False, edgecolor="#991b1b", linewidth=1.1, alpha=0.88))
        ax.text(obstacle.center[0], obstacle.center[1], obstacle.name, ha="center", va="center", fontsize=7, color="#7f1d1d")

    ax.plot(history[:, 0], history[:, 1], color="#93c5fd", linewidth=1.5, alpha=0.55, label="raw history")
    for index, segment in enumerate(segments):
        clearance = tokens[index, 10]
        color = "#dc2626" if clearance < 0.08 else "#2563eb"
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=3.0)
        mid = np.mean(segment, axis=0)
        ax.text(mid[0], mid[1], f"h{index}", fontsize=6.5, color="#0f172a")

    rows, cols = np.where(crossing_hh | near_hh)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if row >= col:
            continue
        a = np.mean(segments[row], axis=0)
        b = np.mean(segments[col], axis=0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#f97316", alpha=0.28, linewidth=0.9)

    rows, cols = np.where(near_ho)
    for row, col in zip(rows.tolist(), cols.tolist()):
        mid = np.mean(segments[row], axis=0)
        center = obstacles[col].center
        ax.plot([mid[0], center[0]], [mid[1], center[1]], color="#ef4444", alpha=0.18, linewidth=0.8)

    ax.set_title("ESC history graph: segmented topological memory")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_relation_heatmaps(graph: dict, obstacles: list[CircleObstacle], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6))

    ho_clearance = graph["history_obstacle"]["clearance"]
    im0 = axes[0].imshow(ho_clearance, aspect="auto", cmap="coolwarm")
    axes[0].set_title("history-obstacle clearance")
    axes[0].set_xlabel("obstacle")
    axes[0].set_ylabel("history segment")
    axes[0].set_xticks(np.arange(len(obstacles)))
    axes[0].set_xticklabels([obstacle.name for obstacle in obstacles], rotation=90, fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    hh_distance = graph["history_history"]["distance"]
    im1 = axes[1].imshow(hh_distance, aspect="auto", cmap="viridis_r")
    axes[1].set_title("history-history distance")
    axes[1].set_xlabel("history segment")
    axes[1].set_ylabel("history segment")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    hh_risk = graph["history_history"]["near"].astype(float) + graph["history_history"]["crossing"].astype(float)
    im2 = axes[2].imshow(hh_risk, aspect="auto", cmap="Reds", vmin=0.0, vmax=2.0)
    axes[2].set_title("history-history near/crossing")
    axes[2].set_xlabel("history segment")
    axes[2].set_ylabel("history segment")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
