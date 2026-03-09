"""
GraphConstructor Component

This module implements the GraphConstructor component that converts binary road masks
into graph representations for pathfinding. The graph uses pixels as nodes and connects
adjacent road pixels with edges weighted by Euclidean distance.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional, Dict, Any
import logging
from skimage.morphology import skeletonize
from scipy import sparse

logger = logging.getLogger(__name__)


class GraphConstructor:
    """
    Converts binary road masks into graph representations for pathfinding.
    
    The GraphConstructor creates a NetworkX graph where:
    - Nodes represent road pixels (mask value = 1)
    - Edges connect adjacent road pixels
    - Edge weights are Euclidean distances between pixels
    
    Attributes:
        connectivity (int): Connectivity mode - 4 or 8 (default: 8)
    """
    
    def __init__(self, connectivity: int = 8):
        """
        Initialize GraphConstructor with specified connectivity.
        
        Args:
            connectivity (int): Connectivity mode - 4 or 8 neighbors
                               4-connectivity: only horizontal and vertical neighbors
                               8-connectivity: includes diagonal neighbors
        
        Raises:
            ValueError: If connectivity is not 4 or 8
        """
        if connectivity not in [4, 8]:
            raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
        
        self.connectivity = connectivity
        logger.info(f"GraphConstructor initialized with {connectivity}-connectivity")
    
    def build_graph(self, road_mask: np.ndarray) -> nx.Graph:
        """
        Build a NetworkX graph from a binary road mask.
        
        Creates a graph where:
        - Each road pixel (mask value = 1) becomes a node at coordinates (x, y)
        - Adjacent road pixels are connected by edges
        - Edge weights are Euclidean distances between pixel centers
        
        Args:
            road_mask (np.ndarray): Binary 2D array where 1 = road, 0 = non-road
        
        Returns:
            nx.Graph: Undirected graph with nodes at road pixel coordinates
                     and edges weighted by Euclidean distance
        
        Raises:
            ValueError: If road_mask is not 2D or contains values other than 0 and 1
            ValueError: If road_mask contains no road pixels
        
        Requirements: 4.1, 4.2, 4.3, 4.4
        """
        # Validate input
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2D array, got shape {road_mask.shape}")
        
        if not np.all(np.isin(road_mask, [0, 1])):
            raise ValueError("road_mask must contain only 0 and 1 values")
        
        if not np.any(road_mask == 1):
            raise ValueError("road_mask must contain at least one road pixel (value = 1)")
        
        height, width = road_mask.shape
        logger.info(f"Building graph from road mask of shape {road_mask.shape}")
        
        # Create empty graph
        graph = nx.Graph()
        
        # Define neighbor offsets based on connectivity
        if self.connectivity == 4:
            # 4-connectivity: only horizontal and vertical neighbors
            directions = [
                (0, -1),   # left
                (0, 1),    # right
                (-1, 0),   # up
                (1, 0)     # down
            ]
        else:  # 8-connectivity
            # 8-connectivity: includes diagonals
            directions = [
                (-1, -1), (-1, 0), (-1, 1),  # top row
                (0, -1),           (0, 1),    # middle row (left, right)
                (1, -1),  (1, 0),  (1, 1)     # bottom row
            ]
        
        # Add nodes for all road pixels
        # Using (x, y) coordinate format where x is horizontal, y is vertical
        node_count = 0
        for y in range(height):
            for x in range(width):
                if road_mask[y, x] == 1:
                    graph.add_node((x, y))
                    node_count += 1
        
        logger.info(f"Added {node_count} nodes to graph")
        
        # Add edges between adjacent road pixels
        edge_count = 0
        for y in range(height):
            for x in range(width):
                if road_mask[y, x] == 1:
                    current_node = (x, y)
                    
                    # Check all neighbors based on connectivity
                    for dy, dx in directions:
                        nx_coord = x + dx
                        ny_coord = y + dy
                        
                        # Check if neighbor is within bounds
                        if 0 <= nx_coord < width and 0 <= ny_coord < height:
                            # Check if neighbor is a road pixel
                            if road_mask[ny_coord, nx_coord] == 1:
                                neighbor_node = (nx_coord, ny_coord)
                                
                                # Only add edge if not already present (undirected graph)
                                if not graph.has_edge(current_node, neighbor_node):
                                    # Compute Euclidean distance as edge weight
                                    weight = np.sqrt(dx * dx + dy * dy)
                                    graph.add_edge(current_node, neighbor_node, weight=weight)
                                    edge_count += 1
        
        logger.info(f"Added {edge_count} edges to graph")
        logger.info(f"Graph construction complete: {graph.number_of_nodes()} nodes, "
                   f"{graph.number_of_edges()} edges")
        
        return graph

    def connect_to_road(self, coord: Tuple[int, int], road_mask: np.ndarray,
                       max_radius: int = 50) -> Optional[Tuple[int, int]]:
        """
        Find the nearest road pixel to a given coordinate within a search radius.

        Searches for the closest road pixel within max_radius distance. If the
        coordinate is out of bounds, returns None.

        Args:
            coord (Tuple[int, int]): Target coordinate (x, y) to connect to road
            road_mask (np.ndarray): Binary 2D array where 1 = road, 0 = non-road
            max_radius (int): Maximum search radius in pixels (default: 50)

        Returns:
            Optional[Tuple[int, int]]: Nearest road pixel coordinate (x, y) if found,
                                      None if no road pixel within max_radius or coord out of bounds

        Requirements: 4.5, 4.6, 10.1
        """
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2D array, got shape {road_mask.shape}")

        if max_radius <= 0:
            raise ValueError(f"max_radius must be positive, got {max_radius}")

        x, y = coord
        height, width = road_mask.shape

        # If coordinate is out of bounds, return None
        if not (0 <= x < width and 0 <= y < height):
            logger.warning(f"Coordinate {coord} is out of bounds for image of size ({width}, {height})")
            return None

        # Check if coordinate is already on a road
        if road_mask[y, x] == 1:
            logger.debug(f"Coordinate {coord} is already on road")
            return coord

        # Search all pixels within max_radius and find the nearest
        min_distance = float('inf')
        nearest_road_pixel = None

        # Search in a square region that encompasses the circular search area
        search_range = int(np.ceil(max_radius))
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                nx_coord = x + dx
                ny_coord = y + dy

                # Check if within image bounds
                if 0 <= nx_coord < width and 0 <= ny_coord < height:
                    # Check if it's a road pixel
                    if road_mask[ny_coord, nx_coord] == 1:
                        # Calculate Euclidean distance
                        distance = np.sqrt(dx * dx + dy * dy)

                        # Only consider pixels within max_radius
                        if distance <= max_radius and distance < min_distance:
                            min_distance = distance
                            nearest_road_pixel = (nx_coord, ny_coord)

        # Return the nearest road pixel found (or None if none found)
        if nearest_road_pixel is not None:
            logger.info(f"Found nearest road pixel {nearest_road_pixel} at distance "
                      f"{min_distance:.2f} from {coord}")
            return nearest_road_pixel
        else:
            logger.warning(f"No road pixel found within {max_radius} pixels of {coord}")
            return None
    def connect_to_graph(self, coord: Tuple[int, int], graph: nx.Graph,
                        max_radius: int = 50) -> Optional[Tuple[int, int]]:
        """
        Find the nearest graph node to a given coordinate within a search radius.

        Searches for the closest node in the graph within max_radius distance.
        This is used after skeletonization to ensure start/goal connect to actual
        graph nodes, not just road pixels that may have been removed.

        Args:
            coord (Tuple[int, int]): Target coordinate (x, y) to connect to graph
            graph (nx.Graph): Road network graph
            max_radius (int): Maximum search radius in pixels (default: 50)

        Returns:
            Optional[Tuple[int, int]]: Nearest graph node coordinate (x, y) if found,
                                      None if no node within max_radius

        Requirements: 4.5, 4.6, 10.1
        """
        if max_radius <= 0:
            raise ValueError(f"max_radius must be positive, got {max_radius}")

        x, y = coord

        # Check if coordinate is already in the graph
        if graph.has_node(coord):
            logger.debug(f"Coordinate {coord} is already in graph")
            return coord

        # Search all nodes in the graph and find the nearest
        min_distance = float('inf')
        nearest_node = None

        for node in graph.nodes():
            nx_coord, ny_coord = node
            dx = nx_coord - x
            dy = ny_coord - y
            distance = np.sqrt(dx * dx + dy * dy)

            # Only consider nodes within max_radius
            if distance <= max_radius and distance < min_distance:
                min_distance = distance
                nearest_node = node

        # Return the nearest node found (or None if none found)
        if nearest_node is not None:
            logger.info(f"Found nearest graph node {nearest_node} at distance "
                      f"{min_distance:.2f} from {coord}")
            return nearest_node
        else:
            logger.warning(f"No graph node found within {max_radius} pixels of {coord}")
            return None


    def add_start_goal_nodes(self, graph: nx.Graph, start: Tuple[int, int],
                            goal: Tuple[int, int], road_mask: np.ndarray,
                            max_radius: int = 50) -> nx.Graph:
        """
        Add start and goal nodes to the graph and connect them to nearest graph nodes.

        If start or goal coordinates are not in the graph, finds the nearest graph node
        within max_radius and connects them. Uses incremental radius expansion
        if no node is found initially (doubles radius up to 4 times).

        Args:
            graph (nx.Graph): Existing road network graph
            start (Tuple[int, int]): Start coordinate (x, y)
            goal (Tuple[int, int]): Goal coordinate (x, y)
            road_mask (np.ndarray): Binary 2D array where 1 = road, 0 = non-road
            max_radius (int): Maximum search radius in pixels (default: 50)

        Returns:
            nx.Graph: Updated graph with start and goal nodes connected

        Raises:
            ValueError: If no graph node found even after radius expansion

        Requirements: 4.5, 4.6, 10.1
        """
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2D array, got shape {road_mask.shape}")

        height, width = road_mask.shape

        # Validate coordinates are within bounds
        start_x, start_y = start
        goal_x, goal_y = goal

        if not (0 <= start_x < width and 0 <= start_y < height):
            raise ValueError(f"Start coordinate {start} is out of bounds for image "
                           f"of size ({width}, {height})")

        if not (0 <= goal_x < width and 0 <= goal_y < height):
            raise ValueError(f"Goal coordinate {goal} is out of bounds for image "
                           f"of size ({width}, {height})")

        logger.info(f"Adding start node {start} and goal node {goal} to graph")

        # Find nearest graph nodes (not just road pixels) for start with incremental expansion
        start_node = self._find_nearest_graph_node(start, graph, max_radius)
        
        # If no node found, try expanding radius progressively (up to 4x original)
        if start_node is None:
            expansion_attempts = [2, 4]  # Multipliers for expansion (up to 4x)
            for multiplier in expansion_attempts:
                expanded_radius = max_radius * multiplier
                logger.warning(f"Expanding search radius for start coordinate {start} to {expanded_radius}")
                start_node = self._find_nearest_graph_node(start, graph, expanded_radius)
                if start_node is not None:
                    break

            if start_node is None:
                max_expanded = max_radius * expansion_attempts[-1]
                raise ValueError(f"No graph node found within {max_expanded} pixels "
                               f"of start coordinate {start}")

        goal_node = self._find_nearest_graph_node(goal, graph, max_radius)
        
        # If no node found, try expanding radius progressively (up to 4x original)
        if goal_node is None:
            expansion_attempts = [2, 4]  # Multipliers for expansion (up to 4x)
            for multiplier in expansion_attempts:
                expanded_radius = max_radius * multiplier
                logger.warning(f"Expanding search radius for goal coordinate {goal} to {expanded_radius}")
                goal_node = self._find_nearest_graph_node(goal, graph, expanded_radius)
                if goal_node is not None:
                    break

            if goal_node is None:
                max_expanded = max_radius * expansion_attempts[-1]
                raise ValueError(f"No graph node found within {max_expanded} pixels "
                               f"of goal coordinate {goal}")

        # Add start and goal nodes to graph if not already present
        if not graph.has_node(start):
            graph.add_node(start)
            logger.debug(f"Added start node {start} to graph")

        if not graph.has_node(goal):
            graph.add_node(goal)
            logger.debug(f"Added goal node {goal} to graph")

        # Connect start to nearest graph node
        if start != start_node:
            distance = np.sqrt((start_node[0] - start[0])**2 +
                             (start_node[1] - start[1])**2)
            graph.add_edge(start, start_node, weight=distance)
            logger.info(f"Connected start {start} to graph node {start_node} "
                       f"with weight {distance:.2f}")

        # Connect goal to nearest graph node
        if goal != goal_node:
            distance = np.sqrt((goal_node[0] - goal[0])**2 +
                             (goal_node[1] - goal[1])**2)
            graph.add_edge(goal, goal_node, weight=distance)
            logger.info(f"Connected goal {goal} to graph node {goal_node} "
                       f"with weight {distance:.2f}")

        logger.info(f"Successfully added and connected start and goal nodes")

        return graph
    
    def _find_nearest_graph_node(self, coord: Tuple[int, int], graph: nx.Graph,
                                 max_radius: int) -> Optional[Tuple[int, int]]:
        """
        Find the nearest node in the graph to a given coordinate.

        Args:
            coord: Target coordinate (x, y)
            graph: Road network graph
            max_radius: Maximum search radius in pixels

        Returns:
            Nearest graph node coordinate if found, None otherwise
        """
        x, y = coord

        # Check if coordinate is already in the graph
        if graph.has_node(coord):
            logger.debug(f"Coordinate {coord} is already in graph")
            return coord

        # Search all nodes in the graph and find the nearest
        min_distance = float('inf')
        nearest_node = None

        for node in graph.nodes():
            nx_coord, ny_coord = node
            dx = nx_coord - x
            dy = ny_coord - y
            distance = np.sqrt(dx * dx + dy * dy)

            # Only consider nodes within max_radius
            if distance <= max_radius and distance < min_distance:
                min_distance = distance
                nearest_node = node

        # Return the nearest node found (or None if none found)
        if nearest_node is not None:
            logger.info(f"Found nearest graph node {nearest_node} at distance "
                      f"{min_distance:.2f} from {coord}")
            return nearest_node
        else:
            logger.warning(f"No graph node found within {max_radius} pixels of {coord}")
            return None

    def skeletonize_mask(self, road_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological skeletonization to reduce road mask to single-pixel width.

        Skeletonization reduces the graph size by converting thick roads to thin
        centerlines while preserving connectivity. This can reduce node count by
        80-90% for typical road networks.

        Args:
            road_mask (np.ndarray): Binary 2D array where 1 = road, 0 = non-road

        Returns:
            np.ndarray: Skeletonized binary mask with same shape as input

        Requirements: 17.3, 17.4, 19.1
        """
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2D array, got shape {road_mask.shape}")

        if not np.all(np.isin(road_mask, [0, 1])):
            raise ValueError("road_mask must contain only 0 and 1 values")

        logger.info(f"Applying skeletonization to road mask of shape {road_mask.shape}")

        # Convert to boolean for skimage
        road_bool = road_mask.astype(bool)

        # Apply skeletonization
        skeleton = skeletonize(road_bool)

        # Convert back to int
        skeleton_mask = skeleton.astype(int)

        original_pixels = np.sum(road_mask)
        skeleton_pixels = np.sum(skeleton_mask)
        reduction_percent = 100 * (1 - skeleton_pixels / original_pixels) if original_pixels > 0 else 0

        logger.info(f"Skeletonization complete: {original_pixels} -> {skeleton_pixels} pixels "
                   f"({reduction_percent:.1f}% reduction)")

        return skeleton_mask

    def remove_isolated_nodes(self, graph: nx.Graph) -> nx.Graph:
        """
        Remove isolated nodes (nodes with no edges) from the graph.

        Isolated nodes don't contribute to pathfinding and can be safely removed
        to reduce memory usage and improve performance.

        Args:
            graph (nx.Graph): Input graph

        Returns:
            nx.Graph: Graph with isolated nodes removed

        Requirements: 19.4
        """
        initial_node_count = graph.number_of_nodes()

        # Find isolated nodes (degree = 0)
        isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]

        # Remove isolated nodes
        graph.remove_nodes_from(isolated_nodes)

        removed_count = len(isolated_nodes)
        final_node_count = graph.number_of_nodes()

        if removed_count > 0:
            logger.info(f"Removed {removed_count} isolated nodes "
                       f"({initial_node_count} -> {final_node_count} nodes)")
        else:
            logger.debug("No isolated nodes to remove")

        return graph

    def to_sparse_matrix(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Convert graph to sparse matrix representation for memory efficiency.

        Sparse matrices are more memory-efficient for large graphs with low
        edge density, which is typical for road networks.

        Args:
            graph (nx.Graph): Input graph

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'adjacency_matrix': scipy.sparse matrix
                - 'node_list': list of nodes in matrix order
                - 'node_count': number of nodes
                - 'edge_count': number of edges

        Requirements: 19.2
        """
        logger.info(f"Converting graph to sparse matrix representation")

        # Get adjacency matrix as sparse matrix
        node_list = list(graph.nodes())
        adjacency_matrix = nx.adjacency_matrix(graph, nodelist=node_list, weight='weight')

        # Calculate memory usage
        dense_size = len(node_list) ** 2 * 8  # 8 bytes per float64
        sparse_size = adjacency_matrix.data.nbytes + adjacency_matrix.indices.nbytes + adjacency_matrix.indptr.nbytes
        memory_reduction = 100 * (1 - sparse_size / dense_size) if dense_size > 0 else 0

        logger.info(f"Sparse matrix created: {len(node_list)} nodes, {graph.number_of_edges()} edges")
        logger.info(f"Memory: {sparse_size / 1024 / 1024:.2f} MB (sparse) vs "
                   f"{dense_size / 1024 / 1024:.2f} MB (dense), "
                   f"{memory_reduction:.1f}% reduction")

        return {
            'adjacency_matrix': adjacency_matrix,
            'node_list': node_list,
            'node_count': len(node_list),
            'edge_count': graph.number_of_edges()
        }

    def prune_large_graph(self, graph: nx.Graph, max_nodes: int = 100000) -> nx.Graph:
        """
        Prune large graphs by removing low-degree nodes from small components.

        For graphs exceeding max_nodes, this method:
        1. Identifies connected components
        2. Keeps the largest component intact
        3. Removes low-degree nodes from smaller components
        4. Continues until node count is below threshold

        Args:
            graph (nx.Graph): Input graph
            max_nodes (int): Maximum number of nodes (default: 100,000)

        Returns:
            nx.Graph: Pruned graph with fewer nodes

        Requirements: 19.3
        """
        initial_node_count = graph.number_of_nodes()

        if initial_node_count <= max_nodes:
            logger.debug(f"Graph has {initial_node_count} nodes, no pruning needed "
                        f"(threshold: {max_nodes})")
            return graph

        logger.warning(f"Graph has {initial_node_count} nodes, exceeds threshold of {max_nodes}. "
                      f"Applying pruning strategy.")

        # Get connected components sorted by size
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        logger.info(f"Found {len(components)} connected components")

        # Keep the largest component intact
        largest_component = components[0]
        logger.info(f"Largest component has {len(largest_component)} nodes (preserved)")

        # Create new graph with largest component
        pruned_graph = graph.subgraph(largest_component).copy()

        # If still too large, remove low-degree nodes from largest component
        if pruned_graph.number_of_nodes() > max_nodes:
            logger.warning(f"Largest component still exceeds threshold, removing low-degree nodes")

            # Sort nodes by degree
            nodes_by_degree = sorted(pruned_graph.nodes(), key=lambda n: pruned_graph.degree(n))

            # Calculate how many nodes need to be removed
            nodes_to_remove_count = pruned_graph.number_of_nodes() - max_nodes
            
            # Remove lowest-degree nodes until under threshold
            nodes_to_remove = []
            max_degree_to_remove = 2  # Start with degree <= 2
            
            # Try progressively higher degree thresholds if needed
            while len(nodes_to_remove) < nodes_to_remove_count and max_degree_to_remove <= 8:
                nodes_to_remove = []
                for node in nodes_by_degree:
                    if len(nodes_to_remove) >= nodes_to_remove_count:
                        break
                    # Remove nodes up to current degree threshold
                    if pruned_graph.degree(node) <= max_degree_to_remove:
                        nodes_to_remove.append(node)
                
                # If we didn't get enough nodes, increase the degree threshold
                if len(nodes_to_remove) < nodes_to_remove_count:
                    max_degree_to_remove += 1

            pruned_graph.remove_nodes_from(nodes_to_remove)
            logger.info(f"Removed {len(nodes_to_remove)} nodes (degree <= {max_degree_to_remove}) from largest component")

        final_node_count = pruned_graph.number_of_nodes()
        reduction_percent = 100 * (1 - final_node_count / initial_node_count)

        logger.info(f"Pruning complete: {initial_node_count} -> {final_node_count} nodes "
                   f"({reduction_percent:.1f}% reduction)")

        return pruned_graph

    def build_optimized_graph(self, road_mask: np.ndarray, 
                             use_skeletonization: bool = False,
                             remove_isolated: bool = True,
                             prune_threshold: int = 100000) -> nx.Graph:
        """
        Build an optimized graph with optional preprocessing and pruning.

        This method combines multiple optimization strategies:
        - Optional skeletonization to reduce node count
        - Isolated node removal
        - Automatic pruning for very large graphs
        - Comprehensive logging of optimization steps

        Args:
            road_mask (np.ndarray): Binary 2D array where 1 = road, 0 = non-road
            use_skeletonization (bool): Apply skeletonization preprocessing (default: False)
            remove_isolated (bool): Remove isolated nodes (default: True)
            prune_threshold (int): Node count threshold for pruning (default: 100,000)

        Returns:
            nx.Graph: Optimized graph representation

        Requirements: 17.3, 17.4, 19.1, 19.2, 19.3, 19.4, 19.5
        """
        logger.info(f"Building optimized graph from road mask of shape {road_mask.shape}")
        logger.info(f"Optimization settings: skeletonization={use_skeletonization}, "
                   f"remove_isolated={remove_isolated}, prune_threshold={prune_threshold}")

        # Step 1: Optional skeletonization
        if use_skeletonization:
            processed_mask = self.skeletonize_mask(road_mask)
        else:
            processed_mask = road_mask
            logger.debug("Skeletonization disabled, using original mask")

        # Step 2: Build base graph
        graph = self.build_graph(processed_mask)

        # Step 3: Remove isolated nodes
        if remove_isolated:
            graph = self.remove_isolated_nodes(graph)

        # Step 4: Prune if too large
        if graph.number_of_nodes() > prune_threshold:
            graph = self.prune_large_graph(graph, max_nodes=prune_threshold)

        # Step 5: Log final statistics
        logger.info(f"Optimized graph complete: {graph.number_of_nodes()} nodes, "
                   f"{graph.number_of_edges()} edges")

        return graph
