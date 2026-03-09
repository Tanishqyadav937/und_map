"""
PathValidator: Verify path validity and compute scores.

This module implements path validation functionality to check that generated
paths satisfy challenge constraints. It verifies waypoints are within bounds,
checks that segments stay on road pixels using Bresenham's line algorithm,
counts violations, computes path length, and calculates scores.

Author: Urban Mission Planning Solution
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class PathValidator:
    """
    Verify path validity according to challenge constraints.
    
    This class validates paths by checking waypoint bounds, verifying segments
    stay on road pixels, counting violations, computing path length, and
    calculating scores using the formula: 1000 - PathLength - 50 × Violations.
    
    Attributes:
        road_mask (np.ndarray): Binary road mask where 1 = road, 0 = non-road
        image_bounds (Tuple[int, int]): Image dimensions (width, height)
    """
    
    def __init__(self, road_mask: np.ndarray, image_bounds: Tuple[int, int]):
        """
        Initialize PathValidator.
        
        Args:
            road_mask: Binary road mask (2D numpy array) where 1 = road, 0 = non-road
            image_bounds: Image dimensions as (width, height)
        
        Raises:
            ValueError: If road_mask is not 2D or image_bounds is invalid.
        
        Preconditions:
            - road_mask is 2D numpy array
            - road_mask contains only 0 and 1 values
            - image_bounds is tuple of two positive integers
        """
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2D array, got {road_mask.ndim}D")
        if len(image_bounds) != 2 or image_bounds[0] <= 0 or image_bounds[1] <= 0:
            raise ValueError(f"image_bounds must be (width, height) with positive values")
        
        self.road_mask = road_mask
        self.image_bounds = image_bounds
        self.width = image_bounds[0]
        self.height = image_bounds[1]
    
    def validate_path(self, path: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Validate path and compute all metrics.
        
        Checks all waypoints are within bounds, verifies all segments stay on
        road pixels, counts violations, computes path length, and calculates score.
        
        Args:
            path: List of coordinate tuples [(x, y), ...]
        
        Returns:
            Dictionary with keys:
                - is_valid (bool): True if violations = 0 and all waypoints in bounds
                - violations (int): Count of off-road or out-of-bounds pixels
                - path_length (float): Sum of Euclidean distances between waypoints
                - score (float): 1000 - path_length - 50 * violations
                - errors (List[str]): Descriptive error messages
        
        Preconditions:
            - path is non-empty list of coordinate tuples
            - All coordinates in path are tuples of integers
        
        Postconditions:
            - Returns dictionary with all required fields
            - is_valid = True if and only if violations = 0 and all waypoints in bounds
            - violations >= 0
            - path_length >= 0
            - score = 1000 - path_length - 50 * violations
        """
        violations = 0
        path_length = 0.0
        errors = []
        
        # Check minimum waypoints
        if len(path) < 2:
            errors.append("Path must have at least 2 waypoints")
            return {
                'is_valid': False,
                'violations': violations,
                'path_length': path_length,
                'score': 0.0,
                'errors': errors
            }
        
        # Validate each waypoint and segment
        for i in range(len(path)):
            point = path[i]
            
            # Check bounds
            if not self._is_in_bounds(point):
                violations += 1
                errors.append(f"Waypoint {i} at {point} is out of bounds")
            
            # Check segment (if not last point)
            if i < len(path) - 1:
                next_point = path[i + 1]
                
                # Compute segment length
                segment_length = self._euclidean_distance(point, next_point)
                path_length += segment_length
                
                # Check if segment stays on road
                segment_violations = self.check_segment_on_road(point, next_point)
                violations += segment_violations
                
                if segment_violations > 0:
                    errors.append(
                        f"Segment {i} from {point} to {next_point} has "
                        f"{segment_violations} off-road/out-of-bounds pixels"
                    )
        
        # Compute score
        score = self.compute_score(path)
        is_valid = (violations == 0)
        
        return {
            'is_valid': is_valid,
            'violations': violations,
            'path_length': path_length,
            'score': score,
            'errors': errors
        }
    
    def check_segment_on_road(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> int:
        """
        Check if segment stays on road pixels using Bresenham's line algorithm.
        
        Uses Bresenham's line algorithm to get all pixels along the segment,
        then counts how many are off-road or out-of-bounds.
        
        Args:
            p1: Start coordinate (x, y)
            p2: End coordinate (x, y)
        
        Returns:
            Count of off-road or out-of-bounds pixels along the segment.
        
        Preconditions:
            - p1 and p2 are tuples of two integers
        
        Postconditions:
            - Returns non-negative integer
            - All pixels along line segment are checked
        """
        violations = 0
        
        # Get all pixels along line segment using Bresenham's algorithm
        pixels = self.bresenham_line(p1, p2)
        
        for pixel in pixels:
            x, y = pixel
            
            # Check if pixel is within bounds
            if not self._is_in_bounds(pixel):
                violations += 1
            else:
                # Check if pixel is on road (use road_mask[y, x] indexing)
                if self.road_mask[y, x] == 0:
                    violations += 1
        
        return violations
    
    def bresenham_line(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Get all pixels along line segment using Bresenham's line algorithm.
        
        Implements Bresenham's line algorithm to find all pixels that should
        be drawn to represent a line from start to end.
        
        Args:
            start: Starting coordinate (x, y)
            end: Ending coordinate (x, y)
        
        Returns:
            List of coordinate tuples along the line from start to end.
        
        Preconditions:
            - start and end are tuples of two integers
        
        Postconditions:
            - Returns list of coordinate tuples
            - First element is start coordinate
            - Last element is end coordinate
            - All pixels form a connected line
        """
        pixels = []
        
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            pixels.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
        
        return pixels
    
    def compute_path_length(self, path: List[Tuple[int, int]]) -> float:
        """
        Compute total path length as sum of Euclidean distances.
        
        Calculates the sum of Euclidean distances between all consecutive
        waypoint pairs in the path.
        
        Args:
            path: List of coordinate tuples [(x, y), ...]
        
        Returns:
            Total path length as a float.
        
        Preconditions:
            - path contains at least 2 coordinate tuples
        
        Postconditions:
            - Returns non-negative float
            - Result is sum of distances between consecutive waypoints
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(path) - 1):
            segment_length = self._euclidean_distance(path[i], path[i + 1])
            total_length += segment_length
        
        return total_length
    
    def compute_score(self, path: List[Tuple[int, int]]) -> float:
        """
        Compute score using formula: 1000 - PathLength - 50 × Violations.
        
        Calculates the path score according to the challenge scoring formula.
        Higher scores are better. Valid paths with zero violations and shorter
        lengths achieve higher scores.
        
        Args:
            path: List of coordinate tuples [(x, y), ...]
        
        Returns:
            Score as a float (can be negative for paths with many violations).
        
        Preconditions:
            - path is non-empty list of coordinate tuples
        
        Postconditions:
            - Returns float value
            - score = 1000 - path_length - 50 * violations
        """
        # Count violations
        violations = 0
        
        # Check waypoint bounds
        for point in path:
            if not self._is_in_bounds(point):
                violations += 1
        
        # Check segments
        for i in range(len(path) - 1):
            segment_violations = self.check_segment_on_road(path[i], path[i + 1])
            violations += segment_violations
        
        # Compute path length
        path_length = self.compute_path_length(path)
        
        # Calculate score
        score = 1000.0 - path_length - 50.0 * violations
        
        return score
    
    def _is_in_bounds(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is within image bounds.
        
        Args:
            point: Coordinate (x, y)
        
        Returns:
            True if point is within bounds, False otherwise.
        """
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height
    
    def _euclidean_distance(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            p1: First coordinate (x, y)
            p2: Second coordinate (x, y)
        
        Returns:
            Euclidean distance as a float.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.sqrt(dx * dx + dy * dy)
