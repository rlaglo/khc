from __future__ import annotations

import heapq
import numpy as np


class Node:
    def __init__(self, position: tuple[int, int, int], g=0, f=0, h=0, parent=None):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h
        self.f = f

    def __lt__(self, other: Node):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f


def path_from_dijkstra(map_points: np.ndarray, grid_size: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    map_points = np.asarray(map_points, dtype=float)
    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise ValueError("map_points must have shape (N, 3)")
    if map_points.shape[0] < 2:
        raise ValueError("map_points must contain at least start and target")

    max_x,max_y,max_z = grid_size

    dijkstra_start_point = map_points[0]
    dijkstra_goal_point = map_points[-1]
    dijkstra_obstacle_points = map_points[1:-1]

    dijkstra_start_point_tup = tuple(np.round(dijkstra_start_point).astype(int))
    dijkstra_goal_point_tup = tuple(np.round(dijkstra_goal_point).astype(int))
    dijkstra_obstacle_points_tup = []

    for point in dijkstra_obstacle_points:
        point_tup = tuple(np.round(point).astype(int))
        dijkstra_obstacle_points_tup.append(point_tup)

    open_list =[]
    closed_set = set()
    best_g = {dijkstra_start_point_tup: 0}
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    start_node = Node(position=dijkstra_start_point_tup, g=0, h=0, f=0)
    heapq.heappush(open_list, start_node)

    while open_list:
        curr_node = heapq.heappop(open_list)
        if curr_node.position in closed_set:
            continue
        closed_set.add(curr_node.position)

        if curr_node.position == dijkstra_goal_point_tup:
            path = []
            while curr_node is not None:
                path.append(curr_node.position)
                curr_node = curr_node.parent
            return np.array(path[::-1], dtype=float) - 0.5

        for dx,dy,dz in directions:
            new_pos = (curr_node.position[0] + dx, \
                       curr_node.position[1] + dy, \
                       curr_node.position[2] + dz)
            if  0<new_pos[0]<=max_x and \
                0<new_pos[1]<=max_y and \
                0<new_pos[2]<=max_z and new_pos not in dijkstra_obstacle_points_tup \
                :
                g = curr_node.g + 1
                if g >= best_g.get(new_pos, 10**9):
                    continue
                best_g[new_pos] = g
                f = g
                new_node = Node(position=new_pos, g=g, h=0, f=f, parent=curr_node)
                heapq.heappush(open_list, new_node)

    raise ValueError("No collision-free path found")
