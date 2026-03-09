"""
Morphological post-processing module for road mask enhancement.

This module provides morphological operations to improve road mask quality
by connecting fragmented roads and reducing noise.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MorphologicalProcessor:
    """
    Applies morphological operations to improve road mask quality.
    
    Morphological operations help connect fragmented road segments and
    reduce noise in binary road masks. The primary operation is morphological
    closing, which fills small gaps in roads.
    """
    
    def __init__(
        self,
        apply_closing: bool = True,
        kernel_size: int = 3,
        iterations: int = 1
    ):
        """
        Initialize morphological processor.
        
        Args:
            apply_closing: Whether to apply morphological closing operation
            kernel_size: Size of the morphological kernel (3 or 5)
            iterations: Number of times to apply the operation
        
        Raises:
            ValueError: If kernel_size is not 3 or 5
            ValueError: If iterations is not positive
        """
        if kernel_size not in [3, 5]:
            raise ValueError(f"kernel_size must be 3 or 5, got {kernel_size}")
        
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        
        self.enable_closing = apply_closing
        self.kernel_size = kernel_size
        self.iterations = iterations
        
        # Create morphological kernel (square structuring element)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
        
        logger.info(
            f"MorphologicalProcessor initialized: "
            f"apply_closing={apply_closing}, "
            f"kernel_size={kernel_size}, "
            f"iterations={iterations}"
        )
    
    def process(self, road_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to road mask.
        
        Args:
            road_mask: Binary road mask (H, W) with values 0 and 1
        
        Returns:
            Processed binary road mask with same shape
        
        Raises:
            ValueError: If road_mask is not 2D binary array
        
        Preconditions:
            - road_mask is 2D numpy array
            - road_mask contains only 0 and 1 values
        
        Postconditions:
            - Returns binary array with same shape as input
            - Fragmented roads are connected (if closing applied)
            - Road structure is preserved
        
        Validates: Requirements 17.1, 17.2, 17.5
        """
        # Validate input
        if road_mask.ndim != 2:
            raise ValueError(
                f"road_mask must be 2D array, got {road_mask.ndim}D"
            )
        
        unique_values = np.unique(road_mask)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"road_mask must be binary (0 and 1), got values: {unique_values}"
            )
        
        # Convert to uint8 for OpenCV operations
        mask_uint8 = road_mask.astype(np.uint8)
        
        # Store original statistics
        original_road_pixels = np.sum(mask_uint8)
        
        # Apply morphological closing if enabled
        if self.enable_closing:
            logger.debug(
                f"Applying morphological closing: "
                f"kernel_size={self.kernel_size}, "
                f"iterations={self.iterations}"
            )
            
            # Morphological closing = dilation followed by erosion
            # This fills small gaps and connects nearby road segments
            processed_mask = cv2.morphologyEx(
                mask_uint8,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.iterations
            )
            
            # Log the effect of the operation
            processed_road_pixels = np.sum(processed_mask)
            pixels_added = processed_road_pixels - original_road_pixels
            
            logger.info(
                f"Morphological closing applied: "
                f"original_pixels={original_road_pixels}, "
                f"processed_pixels={processed_road_pixels}, "
                f"pixels_added={pixels_added}"
            )
        else:
            logger.debug("Morphological closing disabled, returning original mask")
            processed_mask = mask_uint8
        
        # Ensure output is binary
        processed_mask = (processed_mask > 0).astype(np.uint8)
        
        # Validate output
        assert processed_mask.shape == road_mask.shape, \
            "Output shape must match input shape"
        assert np.all(np.isin(np.unique(processed_mask), [0, 1])), \
            "Output must be binary"
        
        return processed_mask
    
    def apply_closing(
        self,
        road_mask: np.ndarray,
        kernel_size: Optional[int] = None,
        iterations: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply morphological closing operation to road mask.
        
        Morphological closing fills small gaps in roads by performing
        dilation followed by erosion. This helps connect fragmented
        road segments.
        
        Args:
            road_mask: Binary road mask (H, W) with values 0 and 1
            kernel_size: Override kernel size (default: use instance setting)
            iterations: Override iterations (default: use instance setting)
        
        Returns:
            Processed binary road mask with same shape
        
        Validates: Requirements 17.1, 17.2
        """
        # Use instance settings if not overridden
        k_size = kernel_size if kernel_size is not None else self.kernel_size
        iters = iterations if iterations is not None else self.iterations
        
        # Validate parameters
        if k_size not in [3, 5]:
            raise ValueError(f"kernel_size must be 3 or 5, got {k_size}")
        if iters <= 0:
            raise ValueError(f"iterations must be positive, got {iters}")
        
        # Create kernel if size differs from instance kernel
        if k_size != self.kernel_size:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (k_size, k_size)
            )
        else:
            kernel = self.kernel
        
        # Convert to uint8
        mask_uint8 = road_mask.astype(np.uint8)
        
        # Apply closing
        closed_mask = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iters
        )
        
        # Ensure binary output
        closed_mask = (closed_mask > 0).astype(np.uint8)
        
        logger.debug(
            f"Closing applied: kernel_size={k_size}, iterations={iters}"
        )
        
        return closed_mask
    
    def get_statistics(
        self,
        original_mask: np.ndarray,
        processed_mask: np.ndarray
    ) -> dict:
        """
        Compute statistics comparing original and processed masks.
        
        Args:
            original_mask: Original binary road mask
            processed_mask: Processed binary road mask
        
        Returns:
            Dictionary with statistics:
                - original_road_pixels: Number of road pixels in original
                - processed_road_pixels: Number of road pixels in processed
                - pixels_added: Number of pixels added by processing
                - pixels_removed: Number of pixels removed by processing
                - road_ratio_original: Ratio of road pixels in original
                - road_ratio_processed: Ratio of road pixels in processed
        """
        original_road = int(np.sum(original_mask))
        processed_road = int(np.sum(processed_mask))
        total_pixels = original_mask.size
        
        pixels_diff = processed_road - original_road
        
        return {
            "original_road_pixels": original_road,
            "processed_road_pixels": processed_road,
            "pixels_added": max(0, pixels_diff),
            "pixels_removed": max(0, -pixels_diff),
            "road_ratio_original": float(original_road / total_pixels),
            "road_ratio_processed": float(processed_road / total_pixels),
        }


def apply_morphological_closing(
    road_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Convenience function to apply morphological closing to a road mask.
    
    Args:
        road_mask: Binary road mask (H, W) with values 0 and 1
        kernel_size: Size of the morphological kernel (3 or 5)
        iterations: Number of times to apply the operation
    
    Returns:
        Processed binary road mask with same shape
    
    Example:
        >>> mask = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        >>> closed = apply_morphological_closing(mask, kernel_size=3)
        >>> # Gap in the middle is filled
    """
    processor = MorphologicalProcessor(
        apply_closing=True,
        kernel_size=kernel_size,
        iterations=iterations
    )
    return processor.process(road_mask)
