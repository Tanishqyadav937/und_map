#!/usr/bin/env python3
"""
Main execution script for Urban Mission Planning Solution.

This script provides a command-line interface for processing satellite images
and generating path solutions. It supports both single image processing and
batch processing of multiple test images.

Usage:
    # Single image processing
    python main.py --image path/to/image.tif --start 100 200 --goal 500 600 --output outputs/solution.json

    # Batch processing
    python main.py --batch --test-dir data/test/sats --coords data/coordinates.json --output-dir outputs/

    # With custom model checkpoint
    python main.py --image path/to/image.tif --start 100 200 --goal 500 600 --checkpoint models/best_model.pth

Author: Urban Mission Planning Solution
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import torch

from src.config import Config
from src.road_segmentation_model import RoadSegmentationModel
from src.solution_generator import SolutionGenerator
from src.logger import setup_logger, get_logger
from src.reproducibility import set_random_seeds


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Urban Mission Planning Solution - Satellite Image Path Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python main.py --image test_001.tif --start 100 200 --goal 500 600 --output solution.json
  
  # Batch process all test images
  python main.py --batch --test-dir data/test/sats --coords coordinates.json --output-dir outputs/
  
  # Use custom model checkpoint
  python main.py --image test_001.tif --start 100 200 --goal 500 600 --checkpoint models/custom.pth
  
  # Enable debug logging
  python main.py --image test_001.tif --start 100 200 --goal 500 600 --log-level DEBUG
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode for multiple images'
    )
    mode_group.add_argument(
        '--image',
        type=str,
        help='Path to single satellite image TIFF file'
    )
    
    # Single image mode arguments
    parser.add_argument(
        '--start',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        help='Start coordinates (x y) for single image mode'
    )
    parser.add_argument(
        '--goal',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        help='Goal coordinates (x y) for single image mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path for single image mode'
    )
    
    # Batch mode arguments
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Directory containing test images for batch mode'
    )
    parser.add_argument(
        '--coords',
        type=str,
        help='JSON file with start/goal coordinates for batch mode'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for batch mode (default: outputs/)'
    )
    
    # Model configuration
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pth',
        help='Path to model checkpoint file (default: models/best_model.pth)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['unet', 'deeplabv3plus'],
        default=Config.MODEL_ARCHITECTURE,
        help=f'Model architecture (default: {Config.MODEL_ARCHITECTURE})'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default=Config.ENCODER_NAME,
        help=f'Encoder backbone (default: {Config.ENCODER_NAME})'
    )
    
    # Processing configuration
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Segmentation threshold (default: 0.5)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=Config.SIMPLIFICATION_EPSILON,
        help=f'Path simplification epsilon (default: {Config.SIMPLIFICATION_EPSILON})'
    )
    
    # Logging configuration
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=Config.LOG_LEVEL,
        help=f'Logging level (default: {Config.LOG_LEVEL})'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=Config.LOG_FILE,
        help=f'Log file path (default: {Config.LOG_FILE})'
    )
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable logging to file (console only)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=Config.RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {Config.RANDOM_SEED})'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file (overrides command-line args)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
    
    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    if args.batch:
        # Batch mode validation
        if not args.test_dir:
            raise ValueError("--test-dir is required for batch mode")
        if not args.coords:
            raise ValueError("--coords is required for batch mode")
        if not os.path.exists(args.test_dir):
            raise ValueError(f"Test directory not found: {args.test_dir}")
        if not os.path.isdir(args.test_dir):
            raise ValueError(f"Test directory path is not a directory: {args.test_dir}")
        if not os.path.exists(args.coords):
            raise ValueError(f"Coordinates file not found: {args.coords}")
    else:
        # Single image mode validation
        if not args.image:
            raise ValueError("--image is required for single image mode")
        if not args.start:
            raise ValueError("--start coordinates are required for single image mode")
        if not args.goal:
            raise ValueError("--goal coordinates are required for single image mode")
        if not args.output:
            raise ValueError("--output is required for single image mode")
        if not os.path.exists(args.image):
            raise ValueError(f"Image file not found: {args.image}")
        if args.start == args.goal:
            raise ValueError("Start and goal coordinates must be different")
    
    # Model checkpoint validation
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Model checkpoint not found: {args.checkpoint}")
    
    # Threshold validation
    if not 0 < args.threshold < 1:
        raise ValueError(f"Threshold must be in range (0, 1), got {args.threshold}")
    
    # Epsilon validation
    if args.epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {args.epsilon}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config file is invalid
        IOError: If config file cannot be read
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load config file {config_path}: {e}")


def initialize_model(args: argparse.Namespace, logger: logging.Logger) -> RoadSegmentationModel:
    """
    Initialize and load the segmentation model.
    
    Args:
        args: Parsed command-line arguments
        logger: Logger instance
    
    Returns:
        Loaded RoadSegmentationModel
    
    Raises:
        RuntimeError: If model initialization or loading fails
    """
    logger.info("="*80)
    logger.info("INITIALIZING MODEL")
    logger.info("="*80)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    try:
        # Initialize model
        model = RoadSegmentationModel(
            architecture=args.architecture,
            encoder_name=args.encoder,
            device=device
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {model.get_model_info()}")
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


def process_single_image(
    args: argparse.Namespace,
    model: RoadSegmentationModel,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a single satellite image.
    
    Args:
        args: Parsed command-line arguments
        model: Loaded segmentation model
        logger: Logger instance
    
    Returns:
        Processing result dictionary
    
    Raises:
        RuntimeError: If processing fails
    """
    logger.info("="*80)
    logger.info("SINGLE IMAGE PROCESSING")
    logger.info("="*80)
    logger.info(f"Image: {args.image}")
    logger.info(f"Start: ({args.start[0]}, {args.start[1]})")
    logger.info(f"Goal: ({args.goal[0]}, {args.goal[1]})")
    logger.info(f"Output: {args.output}")
    logger.info("="*80)
    
    # Update config with command-line arguments
    config = Config.to_dict()
    config['SIMPLIFICATION_EPSILON'] = args.epsilon
    
    # Initialize solution generator
    generator = SolutionGenerator(model, config)
    
    # Process image
    start_coord = tuple(args.start)
    goal_coord = tuple(args.goal)
    
    try:
        result = generator.process_image(args.image, start_coord, goal_coord)
        
        # Extract image ID from filename
        image_id = os.path.splitext(os.path.basename(args.image))[0]
        
        # Generate solution JSON
        generator.generate_solution_json(image_id, result['path'], args.output)
        
        logger.info("="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Solution saved to: {args.output}")
        logger.info(f"Waypoints: {len(result['path'])}")
        logger.info(f"Path length: {result['validation']['path_length']:.2f}")
        logger.info(f"Violations: {result['validation']['violations']}")
        logger.info(f"Score: {result['validation']['score']:.2f}")
        logger.info(f"Valid: {result['validation']['is_valid']}")
        logger.info(f"Total time: {sum(result['timing'].values()):.2f}s")
        logger.info("="*80)
        
        return result
    
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise RuntimeError(f"Image processing failed: {e}")


def process_batch(
    args: argparse.Namespace,
    model: RoadSegmentationModel,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process multiple images in batch mode.
    
    Args:
        args: Parsed command-line arguments
        model: Loaded segmentation model
        logger: Logger instance
    
    Returns:
        Batch processing summary dictionary
    
    Raises:
        RuntimeError: If batch processing fails
    """
    logger.info("="*80)
    logger.info("BATCH PROCESSING MODE")
    logger.info("="*80)
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Coordinates file: {args.coords}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80)
    
    # Update config with command-line arguments
    config = Config.to_dict()
    config['SIMPLIFICATION_EPSILON'] = args.epsilon
    
    # Initialize solution generator
    generator = SolutionGenerator(model, config)
    
    # Process batch
    try:
        summary = generator.batch_process(
            test_images_dir=args.test_dir,
            coordinates_file=args.coords,
            output_dir=args.output_dir
        )
        
        logger.info("="*80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['successful']/summary['total_images']*100:.1f}%")
        logger.info(f"Average time: {summary['average_time']:.2f}s per image")
        logger.info(f"Average score: {summary['average_score']:.2f}")
        logger.info(f"Total time: {summary['total_time']:.2f}s")
        logger.info("="*80)
        
        # Save summary to JSON
        summary_path = os.path.join(args.output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise RuntimeError(f"Batch processing failed: {e}")


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load config file if provided
        if args.config:
            config_dict = load_config_file(args.config)
            Config.update_from_dict(config_dict)
        
        # Setup logging
        log_file = None if args.no_log_file else args.log_file
        setup_logger(level=args.log_level, log_file=log_file)
        logger = get_logger(__name__)
        
        logger.info("="*80)
        logger.info("URBAN MISSION PLANNING SOLUTION")
        logger.info("="*80)
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info("="*80)
        
        # Validate arguments
        validate_arguments(args)
        
        # Set random seeds for reproducibility
        set_random_seeds(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
        
        # Initialize model
        model = initialize_model(args, logger)
        
        # Process based on mode
        if args.batch:
            result = process_batch(args, model, logger)
        else:
            result = process_single_image(args, model, logger)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        logger.info("="*80)
        logger.info("EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info("="*80)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if '--log-level' in sys.argv and sys.argv[sys.argv.index('--log-level') + 1] == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
