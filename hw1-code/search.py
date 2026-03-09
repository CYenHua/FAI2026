# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq
from functools import lru_cache

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def construct_path(parent, current_state):
    path = []
    while current_state is not None:
        if isinstance(current_state, tuple) and isinstance(current_state[0], tuple):
            path.append(current_state[0])
        else:
            path.append(current_state)
        current_state = parent[current_state]
    return path[::-1]

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    queue = [start]
    parent = {start: None}
    while queue:
        current = queue.pop(0)
        if(maze.isObjective(current[0], current[1])):
            return construct_path(parent, current)
        for neighbor in maze.getNeighbors(current[0], current[1]):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    return []

def Manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

@lru_cache(maxsize=None)
def MST(goals):
    if(len(goals) <= 1):
        return 0
    min_weight = 0
    unvisited = list(goals)
    start_node = unvisited.pop(0)

    distance = [Manhattan(start_node, u) for u in unvisited]
    while(unvisited):
        min_edge = min(distance)
        min_idx = distance.index(min_edge)
        new_node = unvisited.pop(min_idx)
        distance.pop(min_idx)
        
        for i, target in enumerate(unvisited):
            d = Manhattan(new_node, target)
            if(d < distance[i]): 
                distance[i] = d
        min_weight += min_edge
    return min_weight


def heuristic(current, goals):
    if not goals: return 0
    dist_to_nearest = min(Manhattan(current, g) for g in goals)
    return dist_to_nearest + MST(goals)


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # start = maze.getStart()
    # goal = maze.getObjectives()[0]

    # prior_queue = [(Manhattan(start, goal), start)]
    # parent = {start: None}
    # cost = {start: 0}

    # while prior_queue:
    #     cur_score, current = heapq.heappop(prior_queue)
    #     if(maze.isObjective(current[0], current[1])):
    #         return construct_path(parent, current)
    #     for neighbor in maze.getNeighbors(current[0], current[1]):
    #         new_cost = cost[current] + 1
    #         if neighbor not in cost or new_cost < cost[neighbor]:
    #             cost[neighbor] = new_cost
    #             parent[neighbor] = current
    #             score = new_cost + Manhattan(neighbor, goal)
    #             heapq.heappush(prior_queue, (score, neighbor))
    # return []
    return astar_multi(maze)
        

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    # return []
    return astar_multi(maze)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    initial_goals = tuple(sorted(maze.getObjectives()))

    start_state = (start, initial_goals)
    queue = [(heuristic(start, initial_goals), start_state)]
    parent = {start_state: None}
    cost = {start_state: 0}
    visited = set()
    while queue:
        _, cur_state = heapq.heappop(queue)
        if cur_state in visited:
            continue
        visited.add(cur_state)
        cur_pos, remaining_goals = cur_state
        if not remaining_goals:
            return construct_path(parent, cur_state)
        for neighbor in maze.getNeighbors(cur_pos[0], cur_pos[1]):
            if neighbor in remaining_goals:
                new_goals = tuple(g for g in remaining_goals if g != neighbor)
            else:
                new_goals = remaining_goals
            next_state = (neighbor, new_goals)
            new_cost = cost[cur_state] + 1
            if(next_state not in cost or new_cost < cost[next_state]):
                cost[next_state] = new_cost
                parent[next_state] = cur_state
                f_score = new_cost + heuristic(neighbor, new_goals)
                heapq.heappush(queue, (f_score, next_state))

    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def astar_single(maze, start, goal):
        queue = [(Manhattan(start, goal), start)]
        parent = {start: None}
        cost = {start: 0}
        visited = set()

        while queue:
            f_score, cur = heapq.heappop(queue)
            if cur in visited:
                continue
            visited.add(cur)
            if(cur == goal):
                return construct_path(parent, cur)
            for neighbor in maze.getNeighbors(cur[0], cur[1]):
                new_cost = cost[cur] + 1
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = cur
                    heapq.heappush(queue, (new_cost + Manhattan(neighbor, goal), neighbor))
        return []

    current_pos = maze.getStart()
    remaining_goals = list(maze.getObjectives())
    path = [current_pos]
    while remaining_goals:
        nearest_goal = min(remaining_goals, key = lambda g: Manhattan(current_pos, g))
        path_to_next = astar_single(maze, current_pos, nearest_goal)
        path.extend(path_to_next[1:])
        current_pos = nearest_goal
        remaining_goals.remove(nearest_goal)
    return path

    

 
