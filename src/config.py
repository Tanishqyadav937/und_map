"""
Configuration module for Urban Mission Planning Solution.
Contains hyperparameters, paths, and system settings.
"""

import os
from typing import Dict, Any


class Config:
    """Main configuration class for the system."""
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUTS_DIR = "outputs"
    
    # Image processing
    MIN_IMAGE_SIZE = 1500  # Updated to match actual dataset dimensions
    MAX_IMAGE_SIZE = 8192
    PIXEL_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    PIXEL_STD = [0.229, 0.224, 0.225]
    
    # Model configuration
    MODEL_ARCHITECTURE = "unet"  # Options: "unet", "deeplabv3plus"
    ENCODER_NAME = "resnet34"  # Encoder backbone
    ENCODER_WEIGHTS = "imagenet"  # Pre-trained weights
    IN_CHANNELS = 3  # RGB
    OUT_CLASSES = 1  # Binary segmentation
    
    # Training hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LOSS_FUNCTION = "dice"  # Options: "bce", "dice", "focal"
    OPTIMIZER = "adam"  # Options: "adam", "sgd", "adamw"
    
    # Device configuration
    DEVICE = "cuda"  # Will fallback to "cpu" if CUDA not available
    
    # Morphological post-processing
    APPLY_MORPHOLOGICAL_CLOSING = True  # Apply closing to connect fragmented roads
    MORPHOLOGICAL_KERNEL_SIZE = 3  # Kernel size for morphological operations (3 or 5)
    MORPHOLOGICAL_ITERATIONS = 1  # Number of iterations for morphological operations
    
    # Graph construction
    CONNECTIVITY = 8  # Options: 4 or 8
    MAX_SEARCH_DISTANCE = 50  # Max distance to search for nearest road pixel
    APPLY_SKELETONIZATION = True
    
    # Pathfinding
    PATHFINDING_ALGORITHM = "astar"
    SIMPLIFICATION_EPSILON = 2.0
    MAX_WAYPOINTS_BEFORE_SIMPLIFICATION = 100
    
    # Validation
    VIOLATION_PENALTY = 50
    BASE_SCORE = 1000
    
    # Performance optimization
    ENABLE_GPU_ACCELERATION = True
    ENABLE_PREPROCESSING_CACHE = True
    CACHE_DIR = ".cache"
    MEMORY_LIMIT_GB = 1.0  # Memory limit per image in GB
    ENABLE_PROFILING = False
    BATCH_INFERENCE_SIZE = 4  # Batch size for multi-image inference
    
    # Reproducibility
    RANDOM_SEED = 42  # Random seed for reproducibility
    ENABLE_DETERMINISTIC = True  # Enable deterministic algorithms
    SAVE_HYPERPARAMETERS = True  # Save hyperparameters with checkpoints
    
    # Logging
    LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "urban_mission_planning.log"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }
    
    @classmethod
    def update_from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration parameters."""
        assert cls.MIN_IMAGE_SIZE > 0, "MIN_IMAGE_SIZE must be positive"
        assert cls.MAX_IMAGE_SIZE >= cls.MIN_IMAGE_SIZE, "MAX_IMAGE_SIZE must be >= MIN_IMAGE_SIZE"
        assert cls.LEARNING_RATE > 0 and cls.LEARNING_RATE < 1, "LEARNING_RATE must be in (0, 1)"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert cls.NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
        assert cls.MORPHOLOGICAL_KERNEL_SIZE in [3, 5], "MORPHOLOGICAL_KERNEL_SIZE must be 3 or 5"
        assert cls.MORPHOLOGICAL_ITERATIONS > 0, "MORPHOLOGICAL_ITERATIONS must be positive"
        assert cls.CONNECTIVITY in [4, 8], "CONNECTIVITY must be 4 or 8"
        assert cls.SIMPLIFICATION_EPSILON > 0, "SIMPLIFICATION_EPSILON must be positive"
        assert cls.VIOLATION_PENALTY >= 0, "VIOLATION_PENALTY must be non-negative"
        assert cls.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"], "Invalid LOG_LEVEL"
