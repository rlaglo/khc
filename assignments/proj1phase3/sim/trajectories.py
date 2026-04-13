from __future__ import annotations

import numpy as np

from .path_planning import path_from_a_star, path_from_dijkstra
from .trajectory_generator import TrajectoryGenerator


MAPS = {
    "Map 1": np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [3.0, 3.0, 1.0],
            [3.0, 7.0, 1.0],
            [1.0, 5.0, 1.0],
            [3.0, 5.0, 1.0],
            [2.0, 7.0, 1.0],
            [2.0, 9.0, 1.0],
        ]
    ),
    "Map 2": np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [2.0, 5.0, 1.0],
            [4.0, 5.0, 1.0],
            [3.0, 7.0, 1.0],
            [4.0, 9.0, 1.0],
        ]
    ),
    "Map z": np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 3.0, 2.0],
            [3.0, 3.0, 1.0],
            [3.0, 8.0, 5.0],
            [3.0, 8.0, 6.0],
            [3.0, 8.0, 8.0],
            [3.0, 9.0, 8.0],
            [3.0, 8.0, 9.0],
            [3.0, 8.0, 7.0],
            [3.0, 6.0, 5.0],
            [2.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [4.0, 6.0, 4.0],
            [4.0, 8.0, 4.0],
            [4.0, 5.0, 1.0],
            [4.0, 5.0, 2.0],
            [4.0, 5.0, 5.0],
            [4.0, 5.0, 8.0],
            [4.0, 7.0, 7.0],
            [4.0, 7.0, 3.0],
            [4.0, 7.0, 8.0],
            [4.0, 6.0, 7.0],
            [4.0, 7.0, 6.0],
            [3.0, 7.0, 7.0],
            [3.0, 7.0, 4.0],
            [3.0, 7.0, 5.0],
            [4.0, 9.0, 6.0],
            [4.0, 9.0, 9.0],
            [4.0, 9.0, 9.0],
            [3.0, 9.0, 7.0],
            [4.0, 10.0, 8.0],
            [4.0, 9.0, 8.0],
        ]
    ),
}


def _map3_points(rng: np.random.Generator) -> np.ndarray:
    points = [[1.0, 1.0, 1.0]]
    seeds_a = rng.integers(1, 36, size=4)
    seeds_b = rng.integers(35, 71, size=4)
    seeds = np.concatenate([seeds_a, seeds_b])
    for seed in seeds:
        z_coord = seed // 35 + 1
        y_coord = (seed % 35) // 5 + 2
        x_coord = (seed % 35) % 5 + 1
        points.append([float(x_coord), float(y_coord), float(z_coord)])
    points.append([5.0, 9.0, 1.0])
    return np.array(points, dtype=float)


def build_map(map_name: str, rng: np.random.Generator | None = None) -> np.ndarray:
    if map_name in MAPS:
        return MAPS[map_name]
    if map_name == "Map 3": # map3는 랜덤/
        rng = rng or np.random.default_rng()
        return _map3_points(rng)
    raise KeyError(f"Unknown map: {map_name}")


def build_path_from_map(map_points: np.ndarray, planner: str = "astar") -> np.ndarray:
    if planner == "astar":
        return path_from_a_star(map_points)
    if planner == "dijkstra":
        return path_from_dijkstra(map_points)
    raise ValueError(f"Unknown planner: {planner}")


def build_path(map_name: str, rng: np.random.Generator | None = None, planner: str = "astar") -> np.ndarray:
    map_points = build_map(map_name, rng)
    return build_path_from_map(map_points, planner)

# astar로 경로 받아서 generator로 넘겨줌
def build_generator_from_map(map_points: np.ndarray, method: str = "jerk", planner: str = "astar") -> TrajectoryGenerator:
    path = build_path_from_map(map_points, planner)
    return TrajectoryGenerator(path, method=method)


def build_generator(map_name: str, method: str = "jerk", rng: np.random.Generator | None = None, planner: str = "astar") -> TrajectoryGenerator:
    map_points = build_map(map_name, rng)
    return build_generator_from_map(map_points, method, planner)


def trajectory_fn_from_generator(generator: TrajectoryGenerator):
    def _trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
        return generator.evaluate(t)

    return _trajectory
