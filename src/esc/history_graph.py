from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    center: np.ndarray
    radius: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CircleObstacle":
        return CircleObstacle(
            name=str(data.get("name", "obstacle")),
            center=np.asarray(data["center"], dtype=float),
            radius=float(data["radius"]),
        )


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {path.shape}")
    if len(path) < 2:
        raise ValueError("points must contain at least two samples")
    sample_count = max(2, int(sample_count))
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total < 1e-12:
        return np.repeat(path[:1], sample_count, axis=0)
    targets = np.linspace(0.0, total, sample_count)
    samples = []
    for target in targets:
        segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment_index = min(segment_index, len(lengths) - 1)
        local_length = lengths[segment_index]
        ratio = 0.0 if local_length < 1e-12 else (target - cumulative[segment_index]) / local_length
        samples.append((1.0 - ratio) * path[segment_index] + ratio * path[segment_index + 1])
    return np.asarray(samples, dtype=float)


def split_polyline(points: np.ndarray, segment_count: int) -> np.ndarray:
    samples = resample_polyline(points, int(segment_count) + 1)
    return np.stack([samples[:-1], samples[1:]], axis=1)


def segment_tokens(segments: np.ndarray, obstacles: list[CircleObstacle], robot_radius: float = 0.03) -> np.ndarray:
    tokens = []
    segment_count = len(segments)
    for index, segment in enumerate(segments):
        start = segment[0]
        end = segment[1]
        delta = end - start
        length = float(np.linalg.norm(delta))
        direction = delta / length if length > 1e-12 else np.zeros(2)
        midpoint = 0.5 * (start + end)
        clearances = [
            point_segment_distance(obstacle.center, start, end) - obstacle.radius - robot_radius
            for obstacle in obstacles
        ]
        min_clearance = float(np.min(clearances)) if clearances else float("inf")
        nearest_obstacle = int(np.argmin(clearances)) if clearances else -1
        tokens.append(
            [
                start[0],
                start[1],
                end[0],
                end[1],
                midpoint[0],
                midpoint[1],
                direction[0],
                direction[1],
                length,
                index / max(1, segment_count - 1),
                min_clearance,
                float(nearest_obstacle),
            ]
        )
    return np.asarray(tokens, dtype=float)


def history_obstacle_relations(
    segments: np.ndarray,
    obstacles: list[CircleObstacle],
    robot_radius: float = 0.03,
) -> dict[str, np.ndarray]:
    distances = np.zeros((len(segments), len(obstacles)), dtype=float)
    clearances = np.zeros_like(distances)
    signed_sides = np.zeros_like(distances)
    local_angle = np.zeros_like(distances)
    near = np.zeros_like(distances, dtype=bool)
    collision = np.zeros_like(distances, dtype=bool)
    for i, segment in enumerate(segments):
        start, end = segment
        direction = end - start
        for j, obstacle in enumerate(obstacles):
            center = obstacle.center
            distances[i, j] = point_segment_distance(center, start, end)
            clearances[i, j] = distances[i, j] - obstacle.radius - robot_radius
            signed_sides[i, j] = signed_side(start, end, center)
            local_angle[i, j] = local_angle_change(start, end, center)
            near[i, j] = clearances[i, j] <= 0.08
            collision[i, j] = clearances[i, j] < 0.0
    return {
        "distance": distances,
        "clearance": clearances,
        "signed_side": signed_sides,
        "local_angle": local_angle,
        "near": near,
        "collision": collision,
    }


def history_history_relations(segments: np.ndarray, near_threshold: float = 0.08) -> dict[str, np.ndarray]:
    count = len(segments)
    distance = np.zeros((count, count), dtype=float)
    crossing = np.zeros((count, count), dtype=bool)
    near = np.zeros((count, count), dtype=bool)
    antiparallel = np.zeros((count, count), dtype=float)
    temporal_gap = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            a, b = segments[i]
            c, d = segments[j]
            distance[i, j] = segment_distance(a, b, c, d)
            is_crossing = abs(i - j) > 1 and segments_intersect(a, b, c, d)
            crossing[i, j] = is_crossing
            near[i, j] = abs(i - j) > 1 and distance[i, j] <= near_threshold
            antiparallel[i, j] = antiparallel_score(b - a, d - c)
            temporal_gap[i, j] = abs(i - j) / max(1, count - 1)
    return {
        "distance": distance,
        "crossing": crossing,
        "near": near,
        "antiparallel": antiparallel,
        "temporal_gap": temporal_gap,
    }


def build_history_graph(
    history: np.ndarray,
    obstacles: list[CircleObstacle],
    segment_count: int = 32,
    robot_radius: float = 0.03,
    near_threshold: float = 0.08,
) -> dict[str, Any]:
    segments = split_polyline(history, segment_count)
    tokens = segment_tokens(segments, obstacles, robot_radius=robot_radius)
    ho = history_obstacle_relations(segments, obstacles, robot_radius=robot_radius)
    hh = history_history_relations(segments, near_threshold=near_threshold)
    return {
        "segment_count": int(len(segments)),
        "segments": segments,
        "tokens": tokens,
        "history_obstacle": ho,
        "history_history": hh,
    }


def fractional_winding_number(points: np.ndarray, center: np.ndarray) -> float:
    path = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    vectors = path - center[None, :]
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
        return float("inf")
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float((angles[-1] - angles[0]) / (2.0 * np.pi))


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, delta) / denom, 0.0, 1.0))
    projection = start + ratio * delta
    return float(np.linalg.norm(point - projection))


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return float(
        min(
            point_segment_distance(a, c, d),
            point_segment_distance(b, c, d),
            point_segment_distance(c, a, b),
            point_segment_distance(d, a, b),
        )
    )


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-9
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return (
        abs(o1) <= eps and on_segment(a, c, b)
        or abs(o2) <= eps and on_segment(a, d, b)
        or abs(o3) <= eps and on_segment(c, a, d)
        or abs(o4) <= eps and on_segment(c, b, d)
    )


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def on_segment(a: np.ndarray, p: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def signed_side(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    delta = end - start
    norm = float(np.linalg.norm(delta))
    if norm < 1e-12:
        return 0.0
    return float(np.cross(delta / norm, point - start))


def local_angle_change(start: np.ndarray, end: np.ndarray, center: np.ndarray) -> float:
    v0 = start - center
    v1 = end - center
    if np.linalg.norm(v0) < 1e-12 or np.linalg.norm(v1) < 1e-12:
        return float("inf")
    a0 = np.arctan2(v0[1], v0[0])
    a1 = np.arctan2(v1[1], v1[0])
    return float(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))


def antiparallel_score(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip((-cosine + 1.0) * 0.5, 0.0, 1.0))


def arrays_to_lists(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: arrays_to_lists(value) for key, value in data.items()}
    if isinstance(data, list):
        return [arrays_to_lists(value) for value in data]
    if isinstance(data, tuple):
        return [arrays_to_lists(value) for value in data]
    if isinstance(data, (np.bool_,)):
        return bool(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    return data
