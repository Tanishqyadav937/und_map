"""
SolutionGenerator: Orchestrate complete pipeline from image to JSON solution.

This module implements the SolutionGenerator component that coordinates all
pipeline stages: image preprocessing, road segmentation, graph construction,
pathfinding, validation, and JSON output generation. It handles batch processing
and provides comprehensive logging of all processing steps.

Author: Urban Mission Planning Solution
"""

import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

from src.image_preprocessor import ImagePreprocessor
from src.road_segmentation_model import RoadSegmentationModel
from src.graph_constructor import GraphConstructor
from src.pathfinding_engine import PathfindingEngine
from src.path_validator import PathValidator
from src.performance_optimizer import PerformanceOptimizer
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class SolutionGenerator:
    """
    Orchestrate the complete pipeline and generate output JSON files.
    
    This class coordinates all pipeline components to process satellite images
    and generate path solutions. It handles:
    - Complete pipeline orchestration
    - Fallback strategies for disconnected networks
    - JSON output generation
    - Batch processing of multiple images
    - Comprehensive logging and timing
    
    Attributes:
        model (RoadSegmentationModel): Trained segmentation model
        preprocessor (ImagePreprocessor): Image preprocessing component
        graph_constructor (GraphConstructor): Graph construction component
        pathfinding_engine (PathfindingEngine): Pathfinding component
        config (Dict[str, Any]): Configuration dictionary
    """
    
    def __init__(self, model: RoadSegmentationModel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SolutionGenerator with trained model and configuration.
        
        Args:
            model: Trained RoadSegmentationModel for road segmentation
            config: Configuration dictionary (uses Config defaults if None)
        
        Preconditions:
            - model is trained and ready for inference
            - config contains valid parameters (if provided)
        
        Postconditions:
            - All pipeline components are initialized
            - Generator is ready to process images
        """
        self.model = model
        self.config = config or Config.to_dict()
        
        # Initialize pipeline components
        self.preprocessor = ImagePreprocessor(
            min_size=self.config.get('MIN_IMAGE_SIZE', Config.MIN_IMAGE_SIZE),
            max_size=self.config.get('MAX_IMAGE_SIZE', Config.MAX_IMAGE_SIZE)
        )
        
        self.graph_constructor = GraphConstructor(
            connectivity=self.config.get('CONNECTIVITY', Config.CONNECTIVITY)
        )
        
        self.pathfinding_engine = PathfindingEngine(
            algorithm=self.config.get('PATHFINDING_ALGORITHM', Config.PATHFINDING_ALGORITHM)
        )
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            enable_gpu=self.config.get('ENABLE_GPU_ACCELERATION', Config.ENABLE_GPU_ACCELERATION),
            enable_cache=self.config.get('ENABLE_PREPROCESSING_CACHE', Config.ENABLE_PREPROCESSING_CACHE),
            cache_dir=self.config.get('CACHE_DIR', Config.CACHE_DIR),
            memory_limit_gb=self.config.get('MEMORY_LIMIT_GB', Config.MEMORY_LIMIT_GB),
            enable_profiling=self.config.get('ENABLE_PROFILING', Config.ENABLE_PROFILING)
        )
        
        logger.info("SolutionGenerator initialized successfully")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Performance optimization enabled: GPU={self.performance_optimizer.metrics['gpu_enabled']}, "
                   f"Cache={self.performance_optimizer.enable_cache}")
    
    def process_image(
        self,
        image_path: str,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Run complete pipeline for a single image.
        
        Orchestrates all pipeline stages:
        1. Load and preprocess image
        2. Segment roads using trained model
        3. Build graph from road mask
        4. Find path using A* algorithm
        5. Simplify path to reduce waypoints
        6. Validate path and compute score
        7. Apply fallback strategies if needed
        
        Args:
            image_path: Path to satellite image TIFF file
            start: Starting coordinate (x, y)
            goal: Goal coordinate (x, y)
        
        Returns:
            Dictionary containing:
                - 'path': List of waypoint coordinates [[x1, y1], [x2, y2], ...]
                - 'validation': Validation results (violations, score, etc.)
                - 'timing': Processing time for each stage
                - 'metadata': Image info and processing details
        
        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If coordinates are invalid
            RuntimeError: If pipeline fails
        
        Preconditions:
            - image_path points to valid TIFF file
            - start and goal are within image bounds
            - start and goal are different coordinates
        
        Postconditions:
            - Returns valid solution with at least 2 waypoints
            - All waypoints are within image bounds
            - Path connects start to goal (possibly with violations)
        
        Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 15.1, 15.2, 15.3
        """
        logger.info("="*80)
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Start: {start}, Goal: {goal}")
        logger.info("="*80)
        
        timing = {}
        
        # Step 1: Load and preprocess image
        logger.info("Step 1: Loading and preprocessing image...")
        start_time = time.time()
        try:
            # Check cache first
            image_tensor = self.performance_optimizer.get_cached_preprocessing(image_path)
            
            if image_tensor is None:
                # Cache miss - preprocess image
                with self.performance_optimizer.profile_operation("preprocessing"):
                    image_tensor = self.preprocessor.preprocess(image_path)
                    # Save to cache
                    self.performance_optimizer.save_cached_preprocessing(image_path, image_tensor)
            else:
                logger.info("Using cached preprocessing result")
            
            timing['preprocessing'] = time.time() - start_time
            logger.info(f"Preprocessing complete in {timing['preprocessing']:.2f}s")
            
            # Check memory after preprocessing
            self.performance_optimizer.check_memory_limit("preprocessing")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")
        
        # Step 2: Segment roads
        logger.info("Step 2: Segmenting roads...")
        start_time = time.time()
        try:
            with self.performance_optimizer.profile_operation("segmentation"):
                road_mask = self.model.predict(image_tensor)
            
            timing['segmentation'] = time.time() - start_time
            road_pixels = np.sum(road_mask)
            road_ratio = np.mean(road_mask)
            logger.info(f"Segmentation complete in {timing['segmentation']:.2f}s")
            logger.info(f"Road pixels: {road_pixels} ({road_ratio*100:.1f}% of image)")
            
            # Check memory after segmentation
            self.performance_optimizer.check_memory_limit("segmentation")
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise RuntimeError(f"Road segmentation failed: {e}")
        
        # Get image dimensions
        height, width = road_mask.shape
        image_bounds = (width, height)
        
        # Validate coordinates are within bounds
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            raise ValueError(f"Start coordinate {start} is out of bounds for image size {image_bounds}")
        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            raise ValueError(f"Goal coordinate {goal} is out of bounds for image size {image_bounds}")
        if start == goal:
            raise ValueError(f"Start and goal coordinates must be different, got {start}")
        
        # Step 3: Build graph from road mask
        logger.info("Step 3: Building road network graph...")
        start_time = time.time()
        try:
            with self.performance_optimizer.profile_operation("graph_construction"):
                # Use optimized graph building with optional skeletonization
                use_skeletonization = self.config.get('APPLY_SKELETONIZATION', Config.APPLY_SKELETONIZATION)
                road_graph = self.graph_constructor.build_optimized_graph(
                    road_mask,
                    use_skeletonization=use_skeletonization,
                    remove_isolated=True,
                    prune_threshold=100000
                )
                
                # Add start and goal nodes
                max_search_distance = self.config.get('MAX_SEARCH_DISTANCE', Config.MAX_SEARCH_DISTANCE)
                road_graph = self.graph_constructor.add_start_goal_nodes(
                    road_graph, start, goal, road_mask, max_radius=max_search_distance
                )
            
            timing['graph_construction'] = time.time() - start_time
            logger.info(f"Graph construction complete in {timing['graph_construction']:.2f}s")
            logger.info(f"Graph: {road_graph.number_of_nodes()} nodes, {road_graph.number_of_edges()} edges")
            
            # Check memory after graph construction
            if not self.performance_optimizer.check_memory_limit("graph_construction"):
                logger.warning("Memory limit exceeded, performing cleanup")
                self.performance_optimizer.cleanup_memory()
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            raise RuntimeError(f"Graph construction failed: {e}")
        
        # Step 4: Find path
        logger.info("Step 4: Finding path using A* algorithm...")
        start_time = time.time()
        try:
            raw_path = self.pathfinding_engine.find_path(road_graph, start, goal)
            timing['pathfinding'] = time.time() - start_time
            
            if raw_path is None:
                logger.warning("No path found between start and goal")
                logger.warning("Applying fallback strategy: direct line path")
                # Fallback: direct line from start to goal
                raw_path = [start, goal]
            else:
                logger.info(f"Path found with {len(raw_path)} waypoints in {timing['pathfinding']:.2f}s")
        except Exception as e:
            logger.error(f"Pathfinding failed: {e}")
            logger.warning("Applying fallback strategy: direct line path")
            raw_path = [start, goal]
            timing['pathfinding'] = time.time() - start_time
        
        # Step 5: Simplify path
        logger.info("Step 5: Simplifying path...")
        start_time = time.time()
        try:
            max_waypoints = self.config.get('MAX_WAYPOINTS_BEFORE_SIMPLIFICATION', 
                                           Config.MAX_WAYPOINTS_BEFORE_SIMPLIFICATION)
            epsilon = self.config.get('SIMPLIFICATION_EPSILON', Config.SIMPLIFICATION_EPSILON)
            
            if len(raw_path) > max_waypoints:
                simplified_path = self.pathfinding_engine.simplify_path(raw_path, epsilon=epsilon)
                logger.info(f"Path simplified: {len(raw_path)} -> {len(simplified_path)} waypoints")
            else:
                simplified_path = raw_path
                logger.info(f"Path has {len(raw_path)} waypoints, no simplification needed")
            
            timing['simplification'] = time.time() - start_time
        except Exception as e:
            logger.warning(f"Path simplification failed: {e}, using raw path")
            simplified_path = raw_path
            timing['simplification'] = time.time() - start_time
        
        # Step 6: Validate path
        logger.info("Step 6: Validating path...")
        start_time = time.time()
        try:
            validator = PathValidator(road_mask, image_bounds)
            validation_result = validator.validate_path(simplified_path)
            timing['validation'] = time.time() - start_time
            
            logger.info(f"Validation complete in {timing['validation']:.2f}s")
            logger.info(f"Path length: {validation_result['path_length']:.2f}")
            logger.info(f"Violations: {validation_result['violations']}")
            logger.info(f"Score: {validation_result['score']:.2f}")
            logger.info(f"Valid: {validation_result['is_valid']}")
            
            if validation_result['errors']:
                logger.warning("Validation errors:")
                for error in validation_result['errors']:
                    logger.warning(f"  - {error}")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Path validation failed: {e}")
        
        # Calculate total processing time
        total_time = sum(timing.values())
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        # Get performance report
        perf_report = self.performance_optimizer.get_performance_report()
        logger.info(f"Performance: Cache hit rate: {perf_report['cache_hit_rate']:.1%}, "
                   f"Memory warnings: {perf_report['memory_warnings']}")
        
        # Convert path to list format for JSON output
        path_list = [[int(x), int(y)] for x, y in simplified_path]
        
        # Prepare result
        result = {
            'path': path_list,
            'validation': validation_result,
            'timing': timing,
            'performance': perf_report,
            'metadata': {
                'image_path': image_path,
                'image_size': image_bounds,
                'start': start,
                'goal': goal,
                'road_pixels': int(road_pixels),
                'road_ratio': float(road_ratio),
                'graph_nodes': road_graph.number_of_nodes(),
                'graph_edges': road_graph.number_of_edges(),
                'waypoint_count': len(simplified_path)
            }
        }
        
        logger.info("="*80)
        logger.info("Processing complete")
        logger.info("="*80)
        
        return result
    
    def generate_solution_json(
        self,
        image_id: str,
        path: List[Tuple[int, int]],
        output_path: str
    ) -> None:
        """
        Generate solution JSON file in required format.
        
        Creates a JSON file with the format:
        {
            "id": "test_xxx",
            "path": [[x1, y1], [x2, y2], ...]
        }
        
        Args:
            image_id: Test image identifier (e.g., "test_001")
            path: List of waypoint coordinates
            output_path: Path where JSON file will be saved
        
        Raises:
            ValueError: If path has fewer than 2 waypoints
            IOError: If file cannot be written
        
        Preconditions:
            - image_id is non-empty string
            - path contains at least 2 waypoints
            - All coordinates are integers
            - output_path is valid file path
        
        Postconditions:
            - JSON file is created at output_path
            - File contains valid JSON with "id" and "path" fields
            - All coordinates are integers
            - Path has at least 2 waypoints
        
        Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
        """
        logger.info(f"Generating solution JSON for {image_id}")
        
        # Validate inputs
        if not image_id:
            raise ValueError("image_id must be non-empty string")
        
        if len(path) < 2:
            raise ValueError(f"Path must have at least 2 waypoints, got {len(path)}")
        
        # Convert path to list format with integer coordinates
        path_list = [[int(x), int(y)] for x, y in path]
        
        # Create solution dictionary
        solution = {
            "id": image_id,
            "path": path_list
        }
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        # Write JSON file
        try:
            with open(output_path, 'w') as f:
                json.dump(solution, f, indent=2)
            logger.info(f"Solution JSON saved to: {output_path}")
        except IOError as e:
            logger.error(f"Failed to write JSON file: {e}")
            raise IOError(f"Failed to write solution JSON to {output_path}: {e}")
    
    def batch_process(
        self,
        test_images_dir: str,
        coordinates_file: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process multiple test images in batch.
        
        Loads test images from directory, reads start/goal coordinates from
        configuration file, processes each image, and generates solution JSON
        files. Continues processing on errors and reports summary statistics.
        
        Args:
            test_images_dir: Directory containing test TIFF images
            coordinates_file: JSON file with start/goal coordinates for each image
            output_dir: Directory where solution JSON files will be saved
        
        Returns:
            Dictionary containing:
                - 'total_images': Total number of images processed
                - 'successful': Number of successful processings
                - 'failed': Number of failed processings
                - 'average_time': Average processing time per image
                - 'average_score': Average score across all images
                - 'results': List of per-image results
        
        Raises:
            FileNotFoundError: If test_images_dir or coordinates_file doesn't exist
            ValueError: If coordinates_file has invalid format
        
        Preconditions:
            - test_images_dir is valid directory path
            - coordinates_file is valid JSON file
            - coordinates_file contains start/goal for each test image
        
        Postconditions:
            - All processable images are processed
            - Solution JSON files are created for successful images
            - Summary statistics are returned
            - Failed images are logged but don't stop processing
        
        Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
        """
        logger.info("="*80)
        logger.info("Starting batch processing")
        logger.info(f"Test images directory: {test_images_dir}")
        logger.info(f"Coordinates file: {coordinates_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*80)
        
        # Validate inputs
        if not os.path.exists(test_images_dir):
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
        if not os.path.exists(coordinates_file):
            raise FileNotFoundError(f"Coordinates file not found: {coordinates_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Load coordinates
        logger.info(f"Loading coordinates from: {coordinates_file}")
        try:
            with open(coordinates_file, 'r') as f:
                coordinates_data = json.load(f)
            logger.info(f"Loaded coordinates for {len(coordinates_data)} images")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in coordinates file: {e}")
        except Exception as e:
            raise IOError(f"Failed to load coordinates file: {e}")
        
        # Get list of test images
        test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.tif') or f.endswith('.tiff')]
        test_images.sort()
        logger.info(f"Found {len(test_images)} test images")
        
        # Process each image
        results = []
        successful = 0
        failed = 0
        total_time = 0.0
        total_score = 0.0
        
        for i, image_filename in enumerate(test_images, 1):
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(test_images_dir, image_filename)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing image {i}/{len(test_images)}: {image_id}")
            logger.info(f"{'='*80}")
            
            # Check if coordinates exist for this image
            if image_id not in coordinates_data:
                logger.warning(f"No coordinates found for {image_id}, skipping")
                failed += 1
                results.append({
                    'image_id': image_id,
                    'status': 'failed',
                    'error': 'No coordinates found'
                })
                continue
            
            # Get start and goal coordinates
            coords = coordinates_data[image_id]
            start = tuple(coords['start'])
            goal = tuple(coords['goal'])
            
            # Process image
            try:
                result = self.process_image(image_path, start, goal)
                
                # Generate solution JSON
                output_path = os.path.join(output_dir, f"{image_id}.json")
                self.generate_solution_json(image_id, result['path'], output_path)
                
                # Update statistics
                successful += 1
                image_time = sum(result['timing'].values())
                total_time += image_time
                total_score += result['validation']['score']
                
                # Store result
                results.append({
                    'image_id': image_id,
                    'status': 'success',
                    'time': image_time,
                    'score': result['validation']['score'],
                    'violations': result['validation']['violations'],
                    'path_length': result['validation']['path_length'],
                    'waypoints': len(result['path'])
                })
                
                logger.info(f"Successfully processed {image_id}")
                logger.info(f"Time: {image_time:.2f}s, Score: {result['validation']['score']:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_id}: {e}")
                failed += 1
                results.append({
                    'image_id': image_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate summary statistics
        average_time = total_time / successful if successful > 0 else 0.0
        average_score = total_score / successful if successful > 0 else 0.0
        
        summary = {
            'total_images': len(test_images),
            'successful': successful,
            'failed': failed,
            'average_time': average_time,
            'average_score': average_score,
            'total_time': total_time,
            'results': results
        }
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['successful']/summary['total_images']*100:.1f}%")
        logger.info(f"Average processing time: {summary['average_time']:.2f}s per image")
        logger.info(f"Average score: {summary['average_score']:.2f}")
        logger.info(f"Total processing time: {summary['total_time']:.2f}s")
        logger.info("="*80)
        
        return summary
