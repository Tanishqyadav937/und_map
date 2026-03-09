"""
Utility module for coordinate validation and conversion.
Ensures consistent coordinate representation throughout the system.

Coordinate System Conventions:
------------------------------
1. Coordinate Format: (x, y) where:
   - x is the horizontal axis (column index)
   - y is the vertical axis (row index)
   - Origin (0, 0) is at the top-left corner

2. Array Indexing: array[row, col] = array[y, x]
   - When accessing numpy arrays: road_mask[y, x]
   - NOT road_mask[x, y] (this is incorrect!)

3. Bounds Checking:
   - Valid x range: [0, width)
   - Valid y range: [0, height)
   - Both x and y must be integers

4. Conversion Rules:
   - Coordinate (x, y) → Array indices [y, x]
   - Array indices [row, col] → Coordinate (col, row)

Examples:
---------
>>> coord = (100, 200)  # x=100, y=200
>>> row, col = coordinate_to_array(coord)  # row=200, col=100
>>> value = road_mask[row, col]  # Correct: road_mask[y, x]
>>> # WRONG: road_mask[coord[0], coord[1]]  # This would be road_mask[x, y]

Validates Requirements: 14.1, 14.2, 14.3, 18.1, 18.2, 18.3, 18.4, 18.5
"""

from typing import Tuple, Optional
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def validate_coordinate(
    coord: Tuple[int, int],
    image_bounds: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Validate that a coordinate is properly formatted and within bounds.
    
    Coordinate system: (x, y) where x is horizontal, y is vertical.
    Origin is at top-left corner (0, 0).
    
    Args:
        coord: Coordinate tuple (x, y)
        image_bounds: Optional (width, height) tuple for bounds checking
    
    Returns:
        True if coordinate is valid, False otherwise
    
    Validates: Requirements 14.1, 14.2, 18.1, 18.2
    """
    try:
        x, y = coord
        
        # Check that coordinates are integers
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            logger.error(f"Coordinates must be integers, got x={type(x)}, y={type(y)}")
            return False
        
        # Check bounds if provided
        if image_bounds is not None:
            width, height = image_bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                logger.error(
                    f"Coordinate ({x}, {y}) out of bounds. "
                    f"Valid range: x=[0, {width}), y=[0, {height})"
                )
                return False
        
        return True
    
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid coordinate format: {coord}. Error: {e}")
        return False


def validate_coordinates_different(
    start: Tuple[int, int],
    goal: Tuple[int, int]
) -> bool:
    """
    Validate that start and goal coordinates are different.
    
    Args:
        start: Start coordinate (x, y)
        goal: Goal coordinate (x, y)
    
    Returns:
        True if coordinates are different, False otherwise
    
    Validates: Requirements 14.3
    """
    if start == goal:
        logger.error(f"Start and goal coordinates must be different. Got: {start}")
        return False
    return True


def array_to_coordinate(array_indices: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert numpy array indices to (x, y) coordinate format.
    
    Array indexing: array[row, col] = array[y, x]
    Coordinate format: (x, y)
    
    Args:
        array_indices: Tuple of (row, col) or (y, x)
    
    Returns:
        Coordinate tuple (x, y)
    
    Example:
        >>> row, col = 200, 100  # array[200, 100]
        >>> x, y = array_to_coordinate((row, col))
        >>> assert x == 100 and y == 200
    
    Validates: Requirements 18.3, 18.4
    """
    row, col = array_indices
    x, y = col, row
    
    # Assertion to ensure correct conversion
    assert x == col, f"X coordinate must equal column index, got x={x}, col={col}"
    assert y == row, f"Y coordinate must equal row index, got y={y}, row={row}"
    
    return (x, y)


def coordinate_to_array(coord: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert (x, y) coordinate to numpy array indices.
    
    Coordinate format: (x, y)
    Array indexing: array[row, col] = array[y, x]
    
    Args:
        coord: Coordinate tuple (x, y)
    
    Returns:
        Array indices tuple (row, col) or (y, x)
    
    Example:
        >>> coord = (100, 200)  # x=100, y=200
        >>> row, col = coordinate_to_array(coord)
        >>> assert row == 200 and col == 100
        >>> # Use with array: value = road_mask[row, col]
    
    Validates: Requirements 18.3, 18.4
    """
    x, y = coord
    row, col = y, x
    
    # Assertion to ensure correct indexing convention
    assert row == y, f"Row index must equal y coordinate, got row={row}, y={y}"
    assert col == x, f"Column index must equal x coordinate, got col={col}, x={x}"
    
    return (row, col)


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Euclidean distance as float
    """
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def validate_image_dimensions(width: int, height: int, min_size: int, max_size: int) -> bool:
    """
    Validate that image dimensions are within acceptable range.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
    
    Returns:
        True if dimensions are valid, False otherwise
    
    Validates: Requirements 1.2
    """
    if width < min_size or height < min_size:
        logger.error(
            f"Image dimensions ({width}x{height}) below minimum size {min_size}x{min_size}"
        )
        return False
    
    if width > max_size or height > max_size:
        logger.error(
            f"Image dimensions ({width}x{height}) exceed maximum size {max_size}x{max_size}"
        )
        return False
    
    return True


def validate_path_format(path: list) -> bool:
    """
    Validate that a path has the correct format for JSON output.
    
    Args:
        path: List of coordinate pairs
    
    Returns:
        True if path format is valid, False otherwise
    
    Validates: Requirements 8.3, 8.4, 8.5
    """
    if not isinstance(path, list):
        logger.error(f"Path must be a list, got {type(path)}")
        return False
    
    if len(path) < 2:
        logger.error(f"Path must have at least 2 waypoints, got {len(path)}")
        return False
    
    for i, waypoint in enumerate(path):
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 2:
            logger.error(f"Waypoint {i} must be a pair of coordinates, got {waypoint}")
            return False
        
        x, y = waypoint
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            logger.error(f"Waypoint {i} coordinates must be integers, got ({type(x)}, {type(y)})")
            return False
    
    return True


def safe_road_mask_access(
    road_mask: np.ndarray,
    coord: Tuple[int, int],
    bounds_check: bool = True
) -> int:
    """
    Safely access road mask value at coordinate with proper indexing.
    
    This function enforces the correct indexing convention: road_mask[y, x]
    
    Args:
        road_mask: Binary road mask array
        coord: Coordinate tuple (x, y)
        bounds_check: Whether to check bounds before access
    
    Returns:
        Road mask value at coordinate (0 or 1)
    
    Raises:
        IndexError: If coordinate is out of bounds
        AssertionError: If indexing convention is violated
    
    Example:
        >>> coord = (100, 200)
        >>> value = safe_road_mask_access(road_mask, coord)
        >>> # Equivalent to: road_mask[200, 100] or road_mask[y, x]
    
    Validates: Requirements 18.4, 18.5
    """
    assert road_mask.ndim == 2, f"Road mask must be 2D array, got {road_mask.ndim}D"
    
    x, y = coord
    height, width = road_mask.shape
    
    if bounds_check:
        if x < 0 or x >= width or y < 0 or y >= height:
            raise IndexError(
                f"Coordinate ({x}, {y}) out of bounds for image size {width}x{height}"
            )
    
    # Use correct indexing: road_mask[y, x] NOT road_mask[x, y]
    value = road_mask[y, x]
    
    # Assertion to catch incorrect usage
    assert 0 <= y < height, f"Y index {y} out of range [0, {height})"
    assert 0 <= x < width, f"X index {x} out of range [0, {width})"
    
    return int(value)


def convert_path_to_json_format(path: list) -> list:
    """
    Convert path to JSON output format with integer coordinates.
    
    Args:
        path: List of coordinate tuples [(x1, y1), (x2, y2), ...]
    
    Returns:
        List of coordinate lists [[x1, y1], [x2, y2], ...]
    
    Validates: Requirements 8.3, 8.4
    """
    json_path = []
    for coord in path:
        x, y = coord
        # Ensure coordinates are Python integers, not numpy integers
        json_path.append([int(x), int(y)])
    
    assert len(json_path) >= 2, "Path must have at least 2 waypoints"
    return json_path
