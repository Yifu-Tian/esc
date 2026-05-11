from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_flow.three_d.collision import Obstacle3D, is_state_collision_free, trajectory_min_clearance
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D, wrap_to_pi


@dataclass(frozen=True)
class RRTConnectConfig:
    max_iterations: int = 3500
    step_size: float = 0.22
    edge_resolution: float = 0.08
    goal_tolerance: float = 0.09
    goal_sample_count: int = 48
    shortcut_attempts: int = 120
    rng_seed: int = 0
    clearance_margin: float = 0.0
    bounds_xy: tuple[tuple[float, float], tuple[float, float]] = ((-2.8, 2.8), (-1.9, 1.9))


@dataclass(frozen=True)
class RRTPlanResult:
    trajectory: np.ndarray
    success: bool
    terminal_error: float
    min_clearance: float
    iterations: int
    message: str


@dataclass
class _Node:
    state: np.ndarray
    parent: int


def plan_rrt_connect_to_goal(
    robot: SimpleMobileManipulator3D,
    start: np.ndarray,
    goal_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
) -> RRTPlanResult:
    rng = np.random.default_rng(config.rng_seed)
    start = _normalize_state(robot, start.copy())
    if not _is_state_valid(robot, start, obstacles, config.clearance_margin):
        return _failure(start, robot, goal_xyz, obstacles, "start state is in collision")

    goal_states = _sample_goal_states(robot, goal_xyz, obstacles, config, rng)
    if not goal_states:
        return _failure(start, robot, goal_xyz, obstacles, "no valid IK goal states")

    tree_a = [_Node(start, -1)]
    tree_b = [_Node(goal_states[0], -1)]
    for goal_state in goal_states[1:]:
        tree_b.append(_Node(goal_state, -1))

    swapped = False
    for iteration in range(config.max_iterations):
        if rng.random() < 0.22:
            sample = goal_states[int(rng.integers(0, len(goal_states)))]
        else:
            sample = _sample_state(robot, config, rng)

        status, new_idx = _extend(tree_a, sample, robot, obstacles, config)
        if status == "trapped":
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped
            continue

        status_b, connect_idx = _connect(tree_b, tree_a[new_idx].state, robot, obstacles, config)
        if status_b == "reached":
            path_a = _path_to_root(tree_a, new_idx)
            path_b = _path_to_root(tree_b, connect_idx)
            if swapped:
                states = path_b + list(reversed(path_a))
            else:
                states = path_a + list(reversed(path_b))
            trajectory = _interpolate_path(np.array(states), config.edge_resolution)
            trajectory = _shortcut_path(robot, trajectory, obstacles, config, rng)
            terminal_error = float(np.linalg.norm(robot.end_effector(trajectory[-1]) - goal_xyz))
            min_clearance = trajectory_min_clearance(robot, trajectory, obstacles)
            success = bool(terminal_error <= config.goal_tolerance and min_clearance >= config.clearance_margin)
            return RRTPlanResult(
                trajectory=trajectory,
                success=success,
                terminal_error=terminal_error,
                min_clearance=float(min_clearance),
                iterations=iteration + 1,
                message="connected",
            )

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    return _failure(start, robot, goal_xyz, obstacles, "max iterations reached")


def _sample_goal_states(
    robot: SimpleMobileManipulator3D,
    goal_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    x_bounds, y_bounds = config.bounds_xy
    max_reach = float(getattr(robot, "max_reach", sum(getattr(robot, "link_lengths", (1.0,)))))
    for _ in range(config.goal_sample_count):
        radius = float(rng.uniform(0.32, min(max_reach - 0.08, 1.35)))
        angle = float(rng.uniform(-np.pi, np.pi))
        base_xy = goal_xyz[:2] - radius * np.array([np.cos(angle), np.sin(angle)])
        if not (x_bounds[0] <= base_xy[0] <= x_bounds[1] and y_bounds[0] <= base_xy[1] <= y_bounds[1]):
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        for elbow_up in [False, True]:
            q = robot.inverse_kinematics_for_goal(base_xy, yaw, goal_xyz, elbow_up=elbow_up)
            state = _normalize_state(robot, np.array([base_xy[0], base_xy[1], yaw, *q], dtype=float))
            terminal_error = np.linalg.norm(robot.end_effector(state) - goal_xyz)
            if terminal_error <= config.goal_tolerance and _is_state_valid(robot, state, obstacles, config.clearance_margin):
                states.append(state)

    # Deterministic fallback around the goal improves repeatability.
    for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
        base_xy = goal_xyz[:2] - 0.95 * np.array([np.cos(angle), np.sin(angle)])
        yaw = wrap_to_pi(angle)
        q = robot.inverse_kinematics_for_goal(base_xy, yaw, goal_xyz, elbow_up=False)
        state = _normalize_state(robot, np.array([base_xy[0], base_xy[1], yaw, *q], dtype=float))
        terminal_error = np.linalg.norm(robot.end_effector(state) - goal_xyz)
        if terminal_error <= config.goal_tolerance and _is_state_valid(robot, state, obstacles, config.clearance_margin):
            states.append(state)
    return states


def _extend(
    tree: list[_Node],
    target: np.ndarray,
    robot: SimpleMobileManipulator3D,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
) -> tuple[str, int]:
    nearest_idx = _nearest(tree, target)
    new_state = _normalize_state(robot, _steer(tree[nearest_idx].state, target, config.step_size))
    if not _is_edge_valid(robot, tree[nearest_idx].state, new_state, obstacles, config):
        return "trapped", nearest_idx
    tree.append(_Node(new_state, nearest_idx))
    if _state_distance(new_state, target) < config.step_size:
        return "reached", len(tree) - 1
    return "advanced", len(tree) - 1


def _connect(
    tree: list[_Node],
    target: np.ndarray,
    robot: SimpleMobileManipulator3D,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
) -> tuple[str, int]:
    last_idx = _nearest(tree, target)
    while True:
        status, idx = _extend(tree, target, robot, obstacles, config)
        if status == "trapped":
            return "trapped", last_idx
        last_idx = idx
        if _state_distance(tree[idx].state, target) <= config.step_size:
            if _is_edge_valid(robot, tree[idx].state, target, obstacles, config):
                tree.append(_Node(_normalize_state(robot, target.copy()), idx))
                return "reached", len(tree) - 1
            return "advanced", idx


def _sample_state(
    robot: SimpleMobileManipulator3D,
    config: RRTConnectConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    x_bounds, y_bounds = config.bounds_xy
    q_lo = np.array([lim[0] for lim in robot.q_limits])
    q_hi = np.array([lim[1] for lim in robot.q_limits])
    state = np.array([
        rng.uniform(*x_bounds),
        rng.uniform(*y_bounds),
        rng.uniform(-np.pi, np.pi),
        *rng.uniform(q_lo, q_hi),
    ])
    return _normalize_state(robot, state)


def _nearest(tree: list[_Node], target: np.ndarray) -> int:
    distances = [_state_distance(node.state, target) for node in tree]
    return int(np.argmin(distances))


def _steer(source: np.ndarray, target: np.ndarray, step_size: float) -> np.ndarray:
    delta = _state_delta(source, target)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return source.copy()
    step = min(step_size, dist)
    return source + delta / dist * step


def _state_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = _state_delta(a, b)
    weights = np.ones_like(delta)
    weights[2:] = 0.35
    return float(np.linalg.norm(weights * delta))


def _state_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    delta[2] = wrap_to_pi(float(delta[2]))
    delta[3] = wrap_to_pi(float(delta[3]))
    return delta


def _normalize_state(robot: SimpleMobileManipulator3D, state: np.ndarray) -> np.ndarray:
    state = state.copy()
    state[2] = wrap_to_pi(float(state[2]))
    if len(state) > 3:
        state[3:] = robot.clip_joints(state[3:])
    return state


def _is_state_valid(
    robot: SimpleMobileManipulator3D,
    state: np.ndarray,
    obstacles: list[Obstacle3D],
    clearance_margin: float,
) -> bool:
    return is_state_collision_free(robot, state, obstacles, clearance_margin)


def _is_edge_valid(
    robot: SimpleMobileManipulator3D,
    a: np.ndarray,
    b: np.ndarray,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
) -> bool:
    delta = _state_delta(a, b)
    steps = max(2, int(np.ceil(float(np.linalg.norm(delta)) / config.edge_resolution)))
    for alpha in np.linspace(0.0, 1.0, steps):
        state = _normalize_state(robot, a + alpha * delta)
        if not _is_state_valid(robot, state, obstacles, config.clearance_margin):
            return False
    return True


def _path_to_root(tree: list[_Node], idx: int) -> list[np.ndarray]:
    out = []
    while idx >= 0:
        out.append(tree[idx].state)
        idx = tree[idx].parent
    return list(reversed(out))


def _interpolate_path(path: np.ndarray, resolution: float) -> np.ndarray:
    if len(path) <= 1:
        return path
    out = [path[0]]
    for i in range(len(path) - 1):
        a = out[-1]
        b = path[i + 1]
        delta = _state_delta(a, b)
        steps = max(1, int(np.ceil(float(np.linalg.norm(delta)) / resolution)))
        for k in range(1, steps + 1):
            out.append(a + delta * (k / steps))
    return np.array(out)


def _shortcut_path(
    robot: SimpleMobileManipulator3D,
    trajectory: np.ndarray,
    obstacles: list[Obstacle3D],
    config: RRTConnectConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    path = trajectory.copy()
    for _ in range(config.shortcut_attempts):
        if len(path) < 4:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j <= i + 2:
            continue
        if _is_edge_valid(robot, path[i], path[j], obstacles, config):
            bridge = _interpolate_path(np.array([path[i], path[j]]), config.edge_resolution)
            path = np.vstack([path[: i + 1], bridge[1:-1], path[j:]])
    return path


def _failure(
    start: np.ndarray,
    robot: SimpleMobileManipulator3D,
    goal_xyz: np.ndarray,
    obstacles: list[Obstacle3D],
    message: str,
) -> RRTPlanResult:
    terminal_error = float(np.linalg.norm(robot.end_effector(start) - goal_xyz))
    return RRTPlanResult(
        trajectory=start[None, :],
        success=False,
        terminal_error=terminal_error,
        min_clearance=float(trajectory_min_clearance(robot, start[None, :], obstacles)),
        iterations=0,
        message=message,
    )
