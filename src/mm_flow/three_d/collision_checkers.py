from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mm_flow.three_d.collision import Obstacle3D, is_state_collision_free, state_clearance
from mm_flow.three_d.kinematics import SimpleMobileManipulator3D


class CollisionChecker(ABC):
    name: str

    @abstractmethod
    def clearance(self, robot: SimpleMobileManipulator3D, state: np.ndarray, obstacles: list[Obstacle3D]) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_state_valid(
        self,
        robot: SimpleMobileManipulator3D,
        state: np.ndarray,
        obstacles: list[Obstacle3D],
        clearance_margin: float = 0.0,
    ) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ProxyCollisionChecker(CollisionChecker):
    name: str = "proxy"

    def clearance(self, robot: SimpleMobileManipulator3D, state: np.ndarray, obstacles: list[Obstacle3D]) -> float:
        return state_clearance(robot, state, obstacles)

    def is_state_valid(
        self,
        robot: SimpleMobileManipulator3D,
        state: np.ndarray,
        obstacles: list[Obstacle3D],
        clearance_margin: float = 0.0,
    ) -> bool:
        return is_state_collision_free(robot, state, obstacles, clearance_margin)


class MeshCollisionChecker(CollisionChecker):
    name = "mesh"

    def clearance(self, robot: SimpleMobileManipulator3D, state: np.ndarray, obstacles: list[Obstacle3D]) -> float:
        raise NotImplementedError("Mesh collision checking is not implemented yet.")

    def is_state_valid(
        self,
        robot: SimpleMobileManipulator3D,
        state: np.ndarray,
        obstacles: list[Obstacle3D],
        clearance_margin: float = 0.0,
    ) -> bool:
        raise NotImplementedError("Mesh collision checking is not implemented yet.")


class SDFCollisionChecker(CollisionChecker):
    name = "sdf"

    def clearance(self, robot: SimpleMobileManipulator3D, state: np.ndarray, obstacles: list[Obstacle3D]) -> float:
        raise NotImplementedError("SDF collision checking is not implemented yet.")

    def is_state_valid(
        self,
        robot: SimpleMobileManipulator3D,
        state: np.ndarray,
        obstacles: list[Obstacle3D],
        clearance_margin: float = 0.0,
    ) -> bool:
        raise NotImplementedError("SDF collision checking is not implemented yet.")


class LearnedCollisionChecker(CollisionChecker):
    name = "learned"

    def clearance(self, robot: SimpleMobileManipulator3D, state: np.ndarray, obstacles: list[Obstacle3D]) -> float:
        raise NotImplementedError("Learned collision checking is not implemented yet.")

    def is_state_valid(
        self,
        robot: SimpleMobileManipulator3D,
        state: np.ndarray,
        obstacles: list[Obstacle3D],
        clearance_margin: float = 0.0,
    ) -> bool:
        raise NotImplementedError("Learned collision checking is not implemented yet.")
