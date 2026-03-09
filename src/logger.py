"""
Logging configuration module for Urban Mission Planning Solution.
Provides centralized logging with configurable levels and formats.

This module implements comprehensive logging functionality including:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Console and file output handlers
- Structured log formatting with timestamps
- Error tracking with stack traces
- Performance timing utilities
- Coordinate adjustment logging

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5
"""

import logging
import sys
import os
import traceback
from typing import Optional, Any, Dict
from datetime import datetime
from src.config import Config


# Global logger registry to track configured loggers
_logger_registry: Dict[str, logging.Logger] = {}


def setup_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Creates a logger with both console and file handlers, formatted with
    timestamps and log levels. Supports configurable log levels and output
    destinations.
    
    Args:
        name: Logger name (typically module name). If None, uses root logger
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to Config.LOG_LEVEL
        log_file: Path to log file. If None, uses Config.LOG_FILE
        console_output: Whether to output logs to console (default: True)
    
    Returns:
        Configured logger instance
    
    Validates: Requirements 15.1, 15.6
    """
    # Use root logger if no name provided
    logger_name = name or 'root'
    
    # Check if logger already configured
    if logger_name in _logger_registry:
        return _logger_registry[logger_name]
    
    # Create logger
    logger = logging.getLogger(logger_name)
    
    # Set level
    log_level = level or Config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter with detailed information
    formatter = logging.Formatter(
        Config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with color support
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation support
    if log_file or Config.LOG_FILE:
        file_path = log_file or Config.LOG_FILE
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Register logger
    _logger_registry[logger_name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Retrieves an existing logger or creates a new one with default settings.
    This is the recommended way to get loggers throughout the application.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    
    Validates: Requirement 15.1
    """
    # Check registry first
    if name in _logger_registry:
        return _logger_registry[name]
    
    # Get logger from logging module
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_image_processing_start(
    logger: logging.Logger,
    image_path: str,
    start: tuple,
    goal: tuple
) -> None:
    """
    Log the start of image processing with coordinates.
    
    Args:
        logger: Logger instance
        image_path: Path to image being processed
        start: Start coordinate (x, y)
        goal: Goal coordinate (x, y)
    
    Validates: Requirement 15.1
    """
    logger.info("="*80)
    logger.info("IMAGE PROCESSING STARTED")
    logger.info("="*80)
    logger.info(f"Image path: {image_path}")
    logger.info(f"Start coordinate: ({start[0]}, {start[1]})")
    logger.info(f"Goal coordinate: ({goal[0]}, {goal[1]})")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


def log_pipeline_stage_timing(
    logger: logging.Logger,
    stage_name: str,
    duration: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log timing information for a pipeline stage.
    
    Args:
        logger: Logger instance
        stage_name: Name of the pipeline stage
        duration: Duration in seconds
        details: Optional dictionary with additional details
    
    Validates: Requirement 15.2
    """
    logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s")
    
    if details:
        for key, value in details.items():
            logger.debug(f"  {key}: {value}")


def log_validation_results(
    logger: logging.Logger,
    path_length: float,
    violations: int,
    score: float,
    is_valid: bool,
    errors: Optional[list] = None
) -> None:
    """
    Log path validation results.
    
    Args:
        logger: Logger instance
        path_length: Total path length
        violations: Number of violations
        score: Computed score
        is_valid: Whether path is valid
        errors: Optional list of error messages
    
    Validates: Requirement 15.3
    """
    logger.info("="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    logger.info(f"Path length: {path_length:.2f} pixels")
    logger.info(f"Violations: {violations}")
    logger.info(f"Score: {score:.2f}")
    logger.info(f"Valid: {is_valid}")
    
    if errors:
        logger.warning(f"Validation errors ({len(errors)}):")
        for error in errors:
            logger.warning(f"  - {error}")
    
    logger.info("="*80)


def log_error_with_traceback(
    logger: logging.Logger,
    error: Exception,
    context: Optional[str] = None
) -> None:
    """
    Log an error with full stack trace.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Optional context description
    
    Validates: Requirement 15.4
    """
    logger.error("="*80)
    logger.error("ERROR OCCURRED")
    logger.error("="*80)
    
    if context:
        logger.error(f"Context: {context}")
    
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.error("Stack trace:")
    logger.error(traceback.format_exc())
    logger.error("="*80)


def log_coordinate_adjustment(
    logger: logging.Logger,
    original: tuple,
    adjusted: tuple,
    reason: str
) -> None:
    """
    Log coordinate adjustments made during processing.
    
    Args:
        logger: Logger instance
        original: Original coordinate (x, y)
        adjusted: Adjusted coordinate (x, y)
        reason: Reason for adjustment
    
    Validates: Requirement 15.5
    """
    logger.warning("="*80)
    logger.warning("COORDINATE ADJUSTMENT")
    logger.warning("="*80)
    logger.warning(f"Original coordinate: ({original[0]}, {original[1]})")
    logger.warning(f"Adjusted coordinate: ({adjusted[0]}, {adjusted[1]})")
    logger.warning(f"Reason: {reason}")
    logger.warning(f"Distance: {((adjusted[0]-original[0])**2 + (adjusted[1]-original[1])**2)**0.5:.2f} pixels")
    logger.warning("="*80)


def log_processing_summary(
    logger: logging.Logger,
    total_time: float,
    waypoints: int,
    score: float,
    success: bool
) -> None:
    """
    Log a summary of processing results.
    
    Args:
        logger: Logger instance
        total_time: Total processing time in seconds
        waypoints: Number of waypoints in path
        score: Final score
        success: Whether processing was successful
    
    Validates: Requirements 15.1, 15.2, 15.3
    """
    logger.info("="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Waypoints: {waypoints}")
    logger.info(f"Score: {score:.2f}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


def log_batch_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    image_id: str,
    elapsed_time: float
) -> None:
    """
    Log progress during batch processing.
    
    Args:
        logger: Logger instance
        current: Current image number
        total: Total number of images
        image_id: Current image identifier
        elapsed_time: Time elapsed for current image
    
    Validates: Requirement 15.2
    """
    progress_pct = (current / total) * 100
    logger.info(f"Progress: {current}/{total} ({progress_pct:.1f}%) - "
               f"{image_id} - {elapsed_time:.2f}s")


def reset_loggers() -> None:
    """
    Reset all configured loggers.
    
    Useful for testing or reconfiguration. Clears the logger registry
    and removes all handlers from registered loggers.
    """
    global _logger_registry
    
    for logger_name, logger in _logger_registry.items():
        logger.handlers.clear()
    
    _logger_registry.clear()
