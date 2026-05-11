from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from mm_flow.three_d.collision import CuboidObstacle, CylinderObstacle, Obstacle3D, SphereObstacle


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def obstacle_to_dict(obstacle: Obstacle3D) -> dict[str, Any]:
    if isinstance(obstacle, SphereObstacle):
        return {"type": "sphere", "center": list(obstacle.center), "radius": obstacle.radius}
    if isinstance(obstacle, CuboidObstacle):
        return {"type": "cuboid", "center": list(obstacle.center), "size": list(obstacle.size)}
    if isinstance(obstacle, CylinderObstacle):
        return {
            "type": "cylinder",
            "center": list(obstacle.center),
            "radius": obstacle.radius,
            "height": obstacle.height,
        }
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")
