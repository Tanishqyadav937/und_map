"""
Reproducibility utilities for deterministic execution.

This module provides functions to ensure reproducible results across multiple runs
by setting random seeds for all random number generators used in the system.

Validates: Requirements 20.1, 20.2, 20.3, 20.5
"""

import torch
import numpy as np
import random
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for all random number generators.
    
    Sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and CUDA)
    
    This ensures deterministic behavior across multiple runs with the same seed.
    
    Args:
        seed: Random seed value (default: 42)
    
    Preconditions:
        - seed is non-negative integer
    
    Postconditions:
        - All random number generators are seeded with the same value
        - Subsequent random operations will be deterministic
        - PyTorch operations will use deterministic algorithms where possible
    
    Validates: Requirements 20.1, 20.2
    
    Example:
        >>> set_random_seeds(42)
        >>> # All random operations will now be deterministic
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    
    logger.info(f"Setting random seeds to {seed} for reproducibility")
    
    # Set Python random seed
    random.seed(seed)
    logger.debug(f"Python random seed set to {seed}")
    
    # Set NumPy random seed
    np.random.seed(seed)
    logger.debug(f"NumPy random seed set to {seed}")
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    logger.debug(f"PyTorch CPU random seed set to {seed}")
    
    # Set PyTorch random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        logger.debug(f"PyTorch CUDA random seed set to {seed}")
    
    # Configure PyTorch to use deterministic algorithms
    # Note: This may impact performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("PyTorch deterministic mode enabled")
    
    logger.info("Random seeds set successfully for reproducible execution")


def enable_deterministic_mode() -> None:
    """
    Enable deterministic mode for PyTorch operations.
    
    Configures PyTorch to use deterministic algorithms wherever possible.
    This may reduce performance but ensures reproducible results.
    
    Preconditions:
        - PyTorch is installed and available
    
    Postconditions:
        - PyTorch operations will be deterministic
        - CUDNN deterministic mode is enabled
        - CUDNN benchmark mode is disabled
    
    Validates: Requirement 20.1
    
    Note:
        Some PyTorch operations do not have deterministic implementations.
        In such cases, PyTorch will raise an error if deterministic mode is required.
    """
    logger.info("Enabling deterministic mode for PyTorch")
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Disable CUDNN benchmark for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("Deterministic mode enabled")


def disable_deterministic_mode() -> None:
    """
    Disable deterministic mode for PyTorch operations.
    
    Restores default PyTorch behavior for better performance.
    Use this when reproducibility is not required.
    
    Postconditions:
        - PyTorch operations may be non-deterministic
        - CUDNN benchmark mode is enabled for better performance
    """
    logger.info("Disabling deterministic mode for PyTorch")
    
    # Disable deterministic algorithms
    torch.use_deterministic_algorithms(False)
    
    # Enable CUDNN benchmark for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    logger.info("Deterministic mode disabled")


def get_random_state() -> Dict[str, Any]:
    """
    Get current state of all random number generators.
    
    Returns:
        Dictionary containing random states for:
            - 'python': Python random module state
            - 'numpy': NumPy random state
            - 'torch_cpu': PyTorch CPU random state
            - 'torch_cuda': PyTorch CUDA random state (if available)
    
    Postconditions:
        - Returns dictionary with all random states
        - States can be restored using set_random_state()
    
    Validates: Requirement 20.2
    
    Example:
        >>> state = get_random_state()
        >>> # Perform some random operations
        >>> set_random_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    logger.debug("Random state captured")
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Restore random number generator states.
    
    Args:
        state: Dictionary containing random states (from get_random_state())
    
    Preconditions:
        - state is valid dictionary from get_random_state()
        - state contains required keys
    
    Postconditions:
        - All random number generators are restored to saved state
        - Subsequent random operations will continue from saved state
    
    Validates: Requirement 20.2
    
    Raises:
        ValueError: If state dictionary is invalid or missing required keys
    """
    required_keys = ['python', 'numpy', 'torch_cpu']
    missing_keys = [key for key in required_keys if key not in state]
    
    if missing_keys:
        raise ValueError(f"Invalid state dictionary, missing keys: {missing_keys}")
    
    logger.debug("Restoring random state")
    
    # Restore Python random state
    random.setstate(state['python'])
    
    # Restore NumPy random state
    np.random.set_state(state['numpy'])
    
    # Restore PyTorch CPU random state
    torch.set_rng_state(state['torch_cpu'])
    
    # Restore PyTorch CUDA random state (if available)
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    
    logger.debug("Random state restored")


def save_hyperparameters(hyperparameters: Dict[str, Any], filepath: str) -> None:
    """
    Save hyperparameters to file for reproducibility.
    
    Saves all hyperparameters and configuration settings to a JSON file
    so that experiments can be reproduced exactly.
    
    Args:
        hyperparameters: Dictionary of hyperparameters and settings
        filepath: Path where hyperparameters will be saved
    
    Preconditions:
        - hyperparameters is valid dictionary
        - filepath is valid file path
    
    Postconditions:
        - Hyperparameters are saved to JSON file
        - File can be loaded to reproduce experiment
    
    Validates: Requirement 20.3
    
    Raises:
        IOError: If file cannot be written
    
    Example:
        >>> hyperparams = {
        ...     'learning_rate': 0.001,
        ...     'batch_size': 4,
        ...     'random_seed': 42
        ... }
        >>> save_hyperparameters(hyperparams, 'config.json')
    """
    import json
    
    logger.info(f"Saving hyperparameters to {filepath}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save hyperparameters as JSON
        with open(filepath, 'w') as f:
            json.dump(hyperparameters, f, indent=2, default=str)
        
        logger.info(f"Hyperparameters saved successfully to {filepath}")
        logger.debug(f"Saved {len(hyperparameters)} hyperparameters")
        
    except IOError as e:
        error_msg = f"Failed to save hyperparameters to {filepath}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)


def load_hyperparameters(filepath: str) -> Dict[str, Any]:
    """
    Load hyperparameters from file.
    
    Args:
        filepath: Path to hyperparameters JSON file
    
    Returns:
        Dictionary of hyperparameters
    
    Preconditions:
        - filepath points to valid JSON file
        - File was created by save_hyperparameters()
    
    Postconditions:
        - Returns dictionary of hyperparameters
        - Can be used to reproduce experiment
    
    Validates: Requirement 20.3
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file contains invalid JSON
    
    Example:
        >>> hyperparams = load_hyperparameters('config.json')
        >>> set_random_seeds(hyperparams['random_seed'])
    """
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")
    
    logger.info(f"Loading hyperparameters from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            hyperparameters = json.load(f)
        
        logger.info(f"Hyperparameters loaded successfully from {filepath}")
        logger.debug(f"Loaded {len(hyperparameters)} hyperparameters")
        
        return hyperparameters
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in hyperparameters file: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Failed to load hyperparameters from {filepath}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)


def ensure_reproducibility(seed: int = 42, enable_deterministic: bool = True) -> None:
    """
    Convenience function to ensure reproducible execution.
    
    Sets random seeds and enables deterministic mode in one call.
    This is the recommended way to enable reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
        enable_deterministic: Whether to enable deterministic algorithms (default: True)
    
    Preconditions:
        - seed is non-negative integer
    
    Postconditions:
        - All random seeds are set
        - Deterministic mode is enabled (if requested)
        - System is ready for reproducible execution
    
    Validates: Requirements 20.1, 20.5
    
    Example:
        >>> ensure_reproducibility(seed=42)
        >>> # All subsequent operations will be deterministic
    """
    logger.info("Ensuring reproducibility")
    
    # Set random seeds
    set_random_seeds(seed)
    
    # Enable deterministic mode if requested
    if enable_deterministic:
        enable_deterministic_mode()
    
    logger.info(f"Reproducibility ensured with seed={seed}, deterministic={enable_deterministic}")


# Document all random number generation in the system
RANDOM_OPERATIONS = {
    'data_augmentation': [
        'Random horizontal flip',
        'Random vertical flip',
        'Random rotation',
        'Random crop'
    ],
    'model_initialization': [
        'Weight initialization (if not using pretrained weights)',
        'Dropout layers during training'
    ],
    'training': [
        'Data shuffling in DataLoader',
        'Batch sampling',
        'Dropout during forward pass'
    ],
    'graph_construction': [
        'No random operations (deterministic)'
    ],
    'pathfinding': [
        'No random operations (deterministic A* algorithm)'
    ]
}


def document_random_operations() -> str:
    """
    Document all random number generation in the system.
    
    Returns:
        String describing all random operations
    
    Validates: Requirement 20.2
    """
    doc = "Random Number Generation Documentation\n"
    doc += "=" * 50 + "\n\n"
    
    for component, operations in RANDOM_OPERATIONS.items():
        doc += f"{component.upper()}:\n"
        for op in operations:
            doc += f"  - {op}\n"
        doc += "\n"
    
    doc += "To ensure reproducibility, call ensure_reproducibility() before any operations.\n"
    doc += "This will set random seeds for Python, NumPy, and PyTorch.\n"
    
    return doc
