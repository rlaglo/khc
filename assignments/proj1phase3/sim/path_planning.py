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
    map_points = np.asarray(map_points, dtype=float)
    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise ValueError("map_points must have shape (N, 3)")
    if map_points.shape[0] < 2:
        raise ValueError("map_points must contain at least start and target")

# def build_path_from_map(map_points: np.ndarray) -> np.ndarray:
#     return path_from_a_star(map_points)
    # 받은 것은 np.darray, 
    # astar
    # "Map 1": np.array(
    #     [
    #         [1.0, 1.0, 1.0],
    #         [1.0, 2.0, 1.0],
    #         [3.0, 3.0, 1.0],
    #         [3.0, 7.0, 1.0],
    #         [1.0, 5.0, 1.0],
    #         [3.0, 5.0, 1.0],
    #         [2.0, 7.0, 1.0],              1,1,1이 시작지점, 2,9,1이 목표지점
    #         [2.0, 9.0, 1.0],              중간 점들 전부 장애물
    #     ]
    # ),


    #전체 바운더리
    max_x,max_y,max_z = grid_size
    
    # TODO: Implement A* algorithm here
        # 1. Parse start, target, and obstacles
    astar_start_point = map_points[0]
    astar_goal_point = map_points[-1]
    astar_obstacle_points = map_points[1:-1]
    
    astar_start_point_tup = tuple(np.round(astar_start_point).astype(int))
    astar_goal_point_tup = tuple(np.round(astar_goal_point).astype(int))
    astar_obstacle_points_tup = []

    for point in astar_obstacle_points:
        point_tup = tuple(np.round(point).astype(int))
        astar_obstacle_points_tup.append(point_tup)
    # 2. Initialize open/closed lists (or queue/visited sets)
    open_list =[]
    closed_set = set()
    #이동방법
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    # 3. Perform A* search
    # 맨하탄 거리로
    start_h = abs(astar_start_point_tup[0] - astar_goal_point_tup[0]) + \
              abs(astar_start_point_tup[1] - astar_goal_point_tup[1]) + \
              abs(astar_start_point_tup[2] - astar_goal_point_tup[2])
    start_node = Node(position=astar_start_point_tup, g=0, h=start_h, f=start_h)
    heapq.heappush(open_list, start_node)

    while open_list:
        curr_node = heapq.heappop(open_list)
        if curr_node.position in closed_set:
            continue
        closed_set.add(curr_node.position)
        # 도달했다면?
        if curr_node.position == astar_goal_point_tup:
            path = []
            while curr_node is not None:
                path.append(curr_node.position)
                curr_node = curr_node.parent
            return np.array(path[::-1], dtype=float) - 0.5
        # 아니라면? 계속 탐색
        for dx,dy,dz in directions:
            new_pos = (curr_node.position[0] + dx, \
                       curr_node.position[1] + dy, \
                       curr_node.position[2] + dz)
            if  0<new_pos[0]<=max_x and \
                0<new_pos[1]<=max_y and \
                0<new_pos[2]<=max_z and new_pos not in astar_obstacle_points_tup \
                :
                g = curr_node.g + 1
                h = abs(new_pos[0] - astar_goal_point_tup[0]) + \
                    abs(new_pos[1] - astar_goal_point_tup[1]) + \
                    abs(new_pos[2] - astar_goal_point_tup[2])
                f= g + h
                new_node = Node(position=new_pos, g=g, h=h, f=f, parent=curr_node)
                heapq.heappush(open_list, new_node)



    # 4. Reconstruct path

        # raise NotImplementedError("path_from_a_star not implemented")

def path_from_dijkstra(map_points: np.ndarray, grid_size: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    """
    Plan a path on a 3D grid using Dijkstra (6-connected).
    Dijkstra is mathematically equivalent to A* with h(n) = 0.
    """
    map_points = np.asarray(map_points, dtype=float)
    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise ValueError("map_points must have shape (N, 3)")
    if map_points.shape[0] < 2:
        raise ValueError("map_points must contain at least start and target")

    max_x, max_y, max_z = grid_size
    
    # 1. Parse start, target, and obstacles
    start_point = map_points[0]
    goal_point = map_points[-1]
    obstacle_points = map_points[1:-1]
    
    start_point_tup = tuple(np.round(start_point).astype(int))
    goal_point_tup = tuple(np.round(goal_point).astype(int))
    
    # O(1) 탐색을 위해 집합(Set)으로 변환
    obstacle_points_set = {tuple(np.round(p).astype(int)) for p in obstacle_points}

    # 2. Initialize open/closed lists
    open_list = []
    closed_set = set()
    
    # 6방향 이동
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    # 3. Perform Dijkstra search (h=0)
    start_node = Node(position=start_point_tup, g=0, h=0, f=0)
    heapq.heappush(open_list, start_node)

    while open_list:
        curr_node = heapq.heappop(open_list)
        
        if curr_node.position in closed_set:
            continue
        closed_set.add(curr_node.position)
        
        # 목적지 도달 시
        if curr_node.position == goal_point_tup:
            path = []
            while curr_node is not None:
                path.append(curr_node.position)
                curr_node = curr_node.parent
            return np.array(path[::-1], dtype=float) - 0.5
            
        # 주변 탐색
        for dx, dy, dz in directions:
            new_pos = (curr_node.position[0] + dx, 
                       curr_node.position[1] + dy, 
                       curr_node.position[2] + dz)
            
            if (0 < new_pos[0] <= max_x and 
                0 < new_pos[1] <= max_y and 
                0 < new_pos[2] <= max_z and 
                new_pos not in obstacle_points_set):
                
                g = curr_node.g + 1
                # Dijkstra에서는 휴리스틱을 사용하지 않으므로 h=0, 즉 f=g
                new_node = Node(position=new_pos, g=g, h=0, f=g, parent=curr_node)
                heapq.heappush(open_list, new_node)

    raise RuntimeError("Valid path not found in the given map using Dijkstra.")