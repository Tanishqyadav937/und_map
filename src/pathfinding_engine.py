"""
PathfindingEngine: Compute optimal paths through road network graphs.

This module implements the A* pathfinding algorithm to find shortest paths
between start and goal coordinates in road network graphs. It uses Euclidean
distance as the heuristic function and supports path simplification using
the Ramer-Douglas-Peucker algorithm.

Author: Urban Mission Planning Solution
"""

import heapq
from typing import List, Optional, Tuple, Dict, Any
import networkx as nx
import numpy as np


class PathfindingEngine:
    """
    Compute optimal paths through road network graphs using A* algorithm.
    
    This class implements pathfinding functionality for road networks represented
    as NetworkX graphs. It uses the A* algorithm with Euclidean distance heuristic
    to find shortest paths and provides path simplification capabilities.
    
    Attributes:
        algorithm (str): Pathfinding algorithm to use (default: "astar")
    """
    
    def __init__(self, algorithm: str = "astar"):
        """
        Initialize PathfindingEngine.
        
        Args:
            algorithm: Pathfinding algorithm to use. Currently only "astar" is supported.
        
        Raises:
            ValueError: If algorithm is not supported.
        """
        if algorithm not in ["astar"]:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Only 'astar' is supported.")
        self.algorithm = algorithm
    
    def find_path(
        self,
        graph: nx.Graph,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path between start and goal using A* algorithm.
        
        Implements the A* pathfinding algorithm with Euclidean distance heuristic.
        Returns the shortest path from start to goal if one exists, otherwise None.
        
        Args:
            graph: NetworkX Graph representing the road network
            start: Starting coordinate (x, y)
            goal: Goal coordinate (x, y)
        
        Returns:
            List of coordinate tuples from start to goal if path exists, None otherwise.
            The first element is the start coordinate and the last is the goal coordinate.
        
        Raises:
            ValueError: If start or goal are not in the graph.
        
        Preconditions:
            - graph is non-empty NetworkX Graph
            - start and goal are tuples of two integers
            - start and goal are nodes in the graph
            - All edge weights are non-negative
        
        Postconditions:
            - Returns shortest path from start to goal if path exists, None otherwise
            - If path exists: first element is start node, last element is goal node
            - Path is shortest path according to edge weights
            - All consecutive coordinates in path are connected by edges
        """
        # Validate inputs
        if not graph.has_node(start):
            raise ValueError(f"Start node {start} not in graph")
        if not graph.has_node(goal):
            raise ValueError(f"Goal node {goal} not in graph")
        
        # Priority queue: (f_score, counter, node)
        # counter ensures FIFO ordering for equal f_scores
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        # Track which nodes are in open_set for efficient lookup
        open_set_nodes = {start}
        
        # came_from[node] = parent node in optimal path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] = cost of cheapest path from start to node
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # f_score[node] = g_score[node] + heuristic(node, goal)
        f_score: Dict[Tuple[int, int], float] = {start: self.compute_heuristic(start, goal)}
        
        while open_set:
            # Get node with lowest f_score
            current_f, _, current = heapq.heappop(open_set)
            open_set_nodes.discard(current)
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = [goal]
                node = goal
                
                while node in came_from:
                    node = came_from[node]
                    path.insert(0, node)
                
                return path
            
            # Explore neighbors
            for neighbor in graph.neighbors(current):
                # Get edge weight
                edge_data = graph.get_edge_data(current, neighbor)
                edge_weight = edge_data.get('weight', 1.0)
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + edge_weight
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.compute_heuristic(neighbor, goal)
                    
                    # Add neighbor to open set if not already there
                    if neighbor not in open_set_nodes:
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        counter += 1
                        open_set_nodes.add(neighbor)
        
        # No path found
        return None
    
    def compute_heuristic(
        self,
        node: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """
        Compute Euclidean distance heuristic for A* algorithm.
        
        Calculates the straight-line (Euclidean) distance between two points.
        This is an admissible heuristic for A* as it never overestimates the
        actual distance.
        
        Args:
            node: Current node coordinate (x, y)
            goal: Goal node coordinate (x, y)
        
        Returns:
            Euclidean distance between node and goal.
        
        Preconditions:
            - node and goal are tuples of two numbers
        
        Postconditions:
            - Returns non-negative float value
            - Result is the straight-line distance between node and goal
        """
        dx = goal[0] - node[0]
        dy = goal[1] - node[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def simplify_path(
        self,
        path: List[Tuple[int, int]],
        epsilon: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        Reduce number of waypoints using Ramer-Douglas-Peucker algorithm.
        
        Simplifies a path by removing waypoints that are within epsilon distance
        of the line segments connecting their neighbors. The first and last
        waypoints are always preserved.
        
        Args:
            path: List of coordinate tuples representing the path
            epsilon: Tolerance threshold for simplification (default: 2.0)
        
        Returns:
            Simplified path with fewer or equal waypoints.
        
        Raises:
            ValueError: If path has fewer than 2 waypoints or epsilon <= 0.
        
        Preconditions:
            - path contains at least 2 coordinate tuples
            - epsilon > 0
            - All coordinates are valid
        
        Postconditions:
            - Returns simplified path with fewer or equal waypoints
            - First and last waypoints are preserved exactly
            - All removed points are within epsilon distance of simplified path
            - Simplified path has at least 2 waypoints
        """
        if len(path) < 2:
            raise ValueError("Path must have at least 2 waypoints")
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        return self._rdp_simplify(path, epsilon)
    
    def _rdp_simplify(
        self,
        path: List[Tuple[int, int]],
        epsilon: float
    ) -> List[Tuple[int, int]]:
        """
        Recursive implementation of Ramer-Douglas-Peucker algorithm.
        
        Args:
            path: List of coordinate tuples
            epsilon: Tolerance threshold
        
        Returns:
            Simplified path.
        """
        if len(path) <= 2:
            return path
        
        # Find point with maximum perpendicular distance from line segment
        start = path[0]
        end = path[-1]
        max_distance = 0.0
        max_index = 0
        
        for i in range(1, len(path) - 1):
            point = path[i]
            distance = self._perpendicular_distance(point, start, end)
            
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursively simplify left and right segments
            left_path = self._rdp_simplify(path[:max_index + 1], epsilon)
            right_path = self._rdp_simplify(path[max_index:], epsilon)
            
            # Combine results (remove duplicate middle point)
            simplified_path = left_path[:-1] + right_path
        else:
            # All points are within epsilon, keep only endpoints
            simplified_path = [start, end]
        
        return simplified_path
    
    def _perpendicular_distance(
        self,
        point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """
        Compute perpendicular distance from point to line segment.
        
        Args:
            point: Point coordinate (x, y)
            line_start: Line segment start coordinate (x, y)
            line_end: Line segment end coordinate (x, y)
        
        Returns:
            Perpendicular distance from point to line segment.
        """
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        # If line segment is a point
        if dx == 0 and dy == 0:
            return self.compute_heuristic(point, line_start)
        
        # Compute projection parameter
        t = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp to [0, 1]
        
        # Find closest point on line segment
        closest_x = line_start[0] + t * dx
        closest_y = line_start[1] + t * dy
        
        # Return distance to closest point
        return self.compute_heuristic(point, (closest_x, closest_y))
