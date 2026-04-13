from __future__ import annotations

import heapq
import numpy as np


def path_from_a_star(map_points: np.ndarray, grid_size: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    """
    Plan a path on a 3D grid using A* (6-connected).

    Args:
        map_points: An (N, 3) array where:
                    - map_points[0] is the start [x, y, z]
                    - map_points[-1] is the target [x, y, z]
                    - map_points[1:-1] are obstacle locations in the grid.
                    All coordinates are floats but correspond to integer grid centers.
                    Note: The grid indices are 1-based (i.e. x in [1, max_x]).
        grid_size: A tuple (max_x, max_y, max_z) defining the dimensions of the grid.

    Returns:
        np.ndarray: An (M, 3) array of waypoints for the path.
                    The waypoints should be the center of the grid cells (i.e., index - 0.5).
                    The path must start with the start position and end with the target position.
    """
    # A* Pre-check: validate input array shape and minimum content.
    map_points = np.asarray(map_points, dtype=float)
    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise ValueError("map_points must have shape (N, 3)")
    if map_points.shape[0] < 2:
        raise ValueError("map_points must contain at least start and target")

    # A* Setup: store 3D grid limits.
    max_x, max_y, max_z = grid_size

    def to_idx(point: np.ndarray) -> tuple[int, int, int]:
        # A* State representation: convert point to integer grid index.
        idx = tuple(int(round(v)) for v in point)
        if len(idx) != 3:
            raise ValueError("Each map point must contain exactly 3 coordinates")
        return idx

    def in_bounds(node: tuple[int, int, int]) -> bool:
        # A* Validity check: keep expansions inside map bounds.
        x, y, z = node
        return 1 <= x <= max_x and 1 <= y <= max_y and 1 <= z <= max_z

    def heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        # A* Heuristic h(n): Manhattan distance (admissible for 6-connected grid).
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    # A* Problem parsing: extract start, goal, and obstacle set.
    start = to_idx(map_points[0])
    goal = to_idx(map_points[-1])
    obstacles = {to_idx(p) for p in map_points[1:-1]}

    # A* Sanity checks: start/goal must be valid and collision-free.
    if not in_bounds(start) or not in_bounds(goal):
        raise ValueError("Start and target must be inside grid bounds")
    if start in obstacles or goal in obstacles:
        raise ValueError("Start and target cannot overlap obstacle cells")

    # A* Trivial case: if start equals goal, return single-node path.
    if start == goal:
        return np.asarray([start], dtype=float) - 0.5

    # A* Neighbor model: 6-connected motion primitives (unit cost each).
    neighbor_dirs = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    # A* Open set: min-heap ordered by f(n) = g(n) + h(n).
    open_heap: list[tuple[int, int, tuple[int, int, int]]] = []
    push_id = 0
    heapq.heappush(open_heap, (heuristic(start, goal), push_id, start))

    # A* Bookkeeping:
    # - came_from: parent pointer for path reconstruction
    # - g_score: best-known path cost from start
    # - closed: fully expanded nodes
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], int] = {start: 0}
    closed: set[tuple[int, int, int]] = set()

    # A* Main loop: repeatedly expand node with the smallest f-score.
    found = False
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        # Skip stale heap entries that were already finalized.
        if current in closed:
            continue

        # Goal test in A*: stop as soon as goal is popped.
        if current == goal:
            found = True
            break

        # Mark current node as expanded.
        closed.add(current)
        current_g = g_score[current]

        # A* Expansion: check all valid 6-connected neighbors.
        for dx, dy, dz in neighbor_dirs:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)

            # Reject invalid, blocked, or already-finalized neighbors.
            if not in_bounds(nxt) or nxt in obstacles or nxt in closed:
                continue

            # Uniform edge cost: each grid move costs 1.
            tentative_g = current_g + 1
            # A* Relaxation: keep only better paths to neighbor.
            if tentative_g < g_score.get(nxt, 10**9):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                push_id += 1
                # Priority key f(n) = g(n) + h(n).
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, push_id, nxt))

    # A* Failure case: no reachable collision-free path exists.
    if not found:
        raise ValueError("No collision-free path found")

    # A* Path reconstruction: backtrack goal -> start using parent pointers.
    idx_path = [goal]
    while idx_path[-1] != start:
        idx_path.append(came_from[idx_path[-1]])
    idx_path.reverse()

    # Output conversion: grid index [i, j, k] -> cell center [i-0.5, j-0.5, k-0.5].
    return np.asarray(idx_path, dtype=float) - 0.5
