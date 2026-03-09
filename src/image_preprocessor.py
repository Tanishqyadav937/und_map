"""
ImagePreprocessor component for loading and validating satellite images.
Handles TIFF image loading, dimension validation, and error handling.
"""

from typing import Tuple
import numpy as np
import torch
from PIL import Image
from src.logger import get_logger
from src.utils import validate_image_dimensions

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Preprocessor for satellite imagery.
    
    Handles loading TIFF files, validating dimensions, and preparing images
    for the segmentation model.
    
    Validates: Requirements 1.1, 1.2, 1.5
    """
    
    def __init__(self, min_size: int = 2048, max_size: int = 8192):
        """
        Initialize ImagePreprocessor with dimension constraints.
        
        Args:
            min_size: Minimum allowed dimension (default: 2048)
            max_size: Maximum allowed dimension (default: 8192)
        """
        self.min_size = min_size
        self.max_size = max_size
        logger.info(
            f"ImagePreprocessor initialized with dimension range: "
            f"{min_size}x{min_size} to {max_size}x{max_size}"
        )
    
    def load_tiff(self, filepath: str) -> np.ndarray:
        """
        Load TIFF file and convert to numpy array with validation.
        
        Args:
            filepath: Path to TIFF image file
        
        Returns:
            Numpy array with shape (H, W, 3) for RGB images
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If image dimensions are invalid or format is incorrect
            IOError: If file is corrupted or unreadable
        
        Validates: Requirements 1.1, 1.2, 1.5
        """
        logger.info(f"Loading TIFF image from: {filepath}")
        
        try:
            # Attempt to open the TIFF file
            with Image.open(filepath) as img:
                # Verify it's a TIFF file
                if img.format != 'TIFF':
                    raise ValueError(
                        f"Expected TIFF format, got {img.format}. "
                        f"File: {filepath}"
                    )
                
                # Get image dimensions
                width, height = img.size
                logger.debug(f"Image dimensions: {width}x{height}")
                
                # Validate dimensions (Requirement 1.2)
                if not validate_image_dimensions(
                    width, height, self.min_size, self.max_size
                ):
                    raise ValueError(
                        f"Image dimensions {width}x{height} outside valid range "
                        f"[{self.min_size}x{self.min_size}, {self.max_size}x{self.max_size}]. "
                        f"File: {filepath}"
                    )
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    logger.warning(
                        f"Converting image from {img.mode} to RGB mode. "
                        f"File: {filepath}"
                    )
                    img = img.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(img, dtype=np.uint8)
                
                # Verify array shape
                if image_array.ndim != 3 or image_array.shape[2] != 3:
                    raise ValueError(
                        f"Expected RGB image with shape (H, W, 3), "
                        f"got shape {image_array.shape}. File: {filepath}"
                    )
                
                logger.info(
                    f"Successfully loaded TIFF image with shape {image_array.shape}"
                )
                
                return image_array
        
        except FileNotFoundError:
            error_msg = f"TIFF file not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        except (IOError, OSError) as e:
            error_msg = (
                f"Failed to read TIFF file (corrupted or unreadable): {filepath}. "
                f"Error: {str(e)}"
            )
            logger.error(error_msg)
            raise IOError(error_msg) from e
        
        except ValueError as e:
            # Re-raise ValueError with context
            logger.error(str(e))
            raise
        
        except Exception as e:
            error_msg = (
                f"Unexpected error loading TIFF file: {filepath}. "
                f"Error: {str(e)}"
            )
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to the range [0, 1].
        
        Args:
            image: Numpy array with shape (H, W, 3) and dtype uint8
        
        Returns:
            Normalized numpy array with shape (H, W, 3) and dtype float32,
            with pixel values in range [0, 1]
        
        Raises:
            ValueError: If image has invalid shape or dtype
        
        Validates: Requirement 1.3
        """
        logger.debug(f"Normalizing image with shape {image.shape}")
        
        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image with shape (H, W, 3), got {image.shape}"
            )
        
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected image with dtype uint8, got {image.dtype}"
            )
        
        # Normalize to [0, 1] by dividing by 255
        normalized = image.astype(np.float32) / 255.0
        
        logger.debug(
            f"Normalized image: min={normalized.min():.4f}, "
            f"max={normalized.max():.4f}"
        )
        
        return normalized
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert normalized numpy array to PyTorch tensor with shape (3, H, W).
        
        Args:
            image: Normalized numpy array with shape (H, W, 3) and dtype float32
        
        Returns:
            PyTorch tensor with shape (3, H, W) and dtype float32
        
        Raises:
            ValueError: If image has invalid shape or dtype
        
        Validates: Requirement 1.4
        """
        logger.debug(f"Converting image to tensor, input shape: {image.shape}")
        
        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image with shape (H, W, 3), got {image.shape}"
            )
        
        if image.dtype != np.float32:
            raise ValueError(
                f"Expected image with dtype float32, got {image.dtype}"
            )
        
        # Transpose from (H, W, 3) to (3, H, W)
        # NumPy uses (H, W, C) format, PyTorch uses (C, H, W) format
        transposed = np.transpose(image, (2, 0, 1))
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(transposed)
        
        logger.debug(f"Converted to tensor with shape {tensor.shape}")
        
        return tensor
    
    def preprocess(self, filepath: str) -> torch.Tensor:
        """
        Complete preprocessing pipeline: load, normalize, and convert to tensor.
        
        This method orchestrates the full preprocessing pipeline:
        1. Load TIFF image from file
        2. Normalize pixel values to [0, 1]
        3. Convert to PyTorch tensor with shape (3, H, W)
        
        Args:
            filepath: Path to TIFF image file
        
        Returns:
            PyTorch tensor with shape (3, H, W) and dtype float32,
            ready for segmentation model input
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If image dimensions are invalid or format is incorrect
            IOError: If file is corrupted or unreadable
        
        Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
        """
        logger.info(f"Starting preprocessing pipeline for: {filepath}")
        
        # Step 1: Load TIFF image
        image_array = self.load_tiff(filepath)
        
        # Step 2: Normalize pixel values
        normalized_image = self.normalize(image_array)
        
        # Step 3: Convert to tensor
        image_tensor = self.to_tensor(normalized_image)
        
        logger.info(
            f"Preprocessing complete. Output tensor shape: {image_tensor.shape}"
        )
        
        return image_tensor
