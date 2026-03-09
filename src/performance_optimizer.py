"""
Performance Optimizer Module

This module implements performance optimizations for the Urban Mission Planning Solution:
- GPU acceleration for segmentation inference
- Batch processing for multiple images
- Memory monitoring and graph pruning
- Image preprocessing caching
- Performance profiling and bottleneck detection

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import torch
import psutil
import gc
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import hashlib
import pickle

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Performance optimization utilities for the pipeline.
    
    Provides:
    - GPU acceleration management
    - Batch processing for multiple images
    - Memory monitoring and cleanup
    - Preprocessing cache management
    - Performance profiling
    """
    
    def __init__(
        self,
        enable_gpu: bool = True,
        enable_cache: bool = True,
        cache_dir: str = ".cache",
        memory_limit_gb: float = 1.0,
        enable_profiling: bool = False
    ):
        """
        Initialize PerformanceOptimizer.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_cache: Enable preprocessing cache
            cache_dir: Directory for cache files
            memory_limit_gb: Memory limit in GB per image
            enable_profiling: Enable performance profiling
        """
        self.enable_gpu = enable_gpu
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.enable_profiling = enable_profiling
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")
        
        # Detect GPU availability
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Performance metrics
        self.metrics = {
            'gpu_enabled': torch.cuda.is_available() and self.enable_gpu,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_warnings': 0,
            'profiling_data': []
        }
        
        logger.info("PerformanceOptimizer initialized")
    
    def _detect_device(self) -> torch.device:
        """
        Detect and return the best available device.
        
        Returns:
            torch.device: CUDA device if available and enabled, else CPU
        
        Validates: Requirement 11.4
        """
        if self.enable_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
            return device
        else:
            if self.enable_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but not available, falling back to CPU")
            return torch.device("cpu")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB:
                - 'ram_used': System RAM used
                - 'ram_available': System RAM available
                - 'ram_percent': RAM usage percentage
                - 'gpu_used': GPU memory used (if available)
                - 'gpu_total': Total GPU memory (if available)
        
        Validates: Requirement 11.5
        """
        memory_info = {
            'ram_used': psutil.virtual_memory().used / 1024**2,
            'ram_available': psutil.virtual_memory().available / 1024**2,
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_used'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            memory_info['gpu_percent'] = (memory_info['gpu_used'] / memory_info['gpu_total']) * 100
        
        return memory_info
    
    def check_memory_limit(self, operation: str = "operation") -> bool:
        """
        Check if memory usage is within limits.
        
        Args:
            operation: Name of operation being checked
        
        Returns:
            True if within limits, False if exceeding
        
        Validates: Requirement 11.5, 11.6
        """
        memory_info = self.get_memory_usage()
        
        # Check RAM usage
        if memory_info['ram_used'] > self.memory_limit_bytes / 1024**2:
            logger.warning(
                f"Memory usage ({memory_info['ram_used']:.0f} MB) exceeds limit "
                f"({self.memory_limit_bytes / 1024**2:.0f} MB) during {operation}"
            )
            self.metrics['memory_warnings'] += 1
            return False
        
        # Check GPU memory if available
        if torch.cuda.is_available() and 'gpu_percent' in memory_info:
            if memory_info['gpu_percent'] > 90:
                logger.warning(
                    f"GPU memory usage ({memory_info['gpu_percent']:.1f}%) is high during {operation}"
                )
                self.metrics['memory_warnings'] += 1
                return False
        
        return True
    
    def cleanup_memory(self) -> None:
        """
        Perform memory cleanup operations.
        
        Clears Python garbage collector and CUDA cache if available.
        
        Validates: Requirement 11.6
        """
        logger.debug("Performing memory cleanup")
        
        # Python garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    
    def _get_cache_key(self, image_path: str) -> str:
        """
        Generate cache key for an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            MD5 hash of image path and modification time
        """
        path = Path(image_path)
        mtime = path.stat().st_mtime if path.exists() else 0
        key_str = f"{image_path}_{mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_preprocessing(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached preprocessed image tensor.
        
        Args:
            image_path: Path to original image
        
        Returns:
            Cached tensor if available, None otherwise
        
        Validates: Requirement 11.4 (optional caching)
        """
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    tensor = pickle.load(f)
                self.metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for {image_path}")
                return tensor
            except Exception as e:
                logger.warning(f"Failed to load cache for {image_path}: {e}")
                self.metrics['cache_misses'] += 1
                return None
        else:
            self.metrics['cache_misses'] += 1
            return None
    
    def save_cached_preprocessing(self, image_path: str, tensor: torch.Tensor) -> None:
        """
        Save preprocessed image tensor to cache.
        
        Args:
            image_path: Path to original image
            tensor: Preprocessed tensor to cache
        
        Validates: Requirement 11.4 (optional caching)
        """
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(tensor, f)
            logger.debug(f"Cached preprocessing for {image_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {image_path}: {e}")
    
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling operations.
        
        Usage:
            with optimizer.profile_operation("segmentation"):
                # ... operation code ...
        
        Args:
            operation_name: Name of operation being profiled
        
        Validates: Requirement 11.6 (profiling)
        """
        return ProfileContext(self, operation_name)
    
    def batch_predict(
        self,
        model,
        image_tensors: List[torch.Tensor],
        batch_size: int = 4
    ) -> List[np.ndarray]:
        """
        Perform batch inference on multiple images.
        
        Processes multiple images in batches to improve GPU utilization
        and throughput.
        
        Args:
            model: Segmentation model with predict method
            image_tensors: List of preprocessed image tensors
            batch_size: Number of images to process per batch
        
        Returns:
            List of binary road masks
        
        Validates: Requirement 11.4 (batch processing)
        """
        if not image_tensors:
            return []
        
        logger.info(f"Batch processing {len(image_tensors)} images with batch_size={batch_size}")
        
        results = []
        
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i + batch_size]
            
            # Check memory before processing batch
            if not self.check_memory_limit(f"batch_{i//batch_size}"):
                logger.warning("Memory limit reached, performing cleanup")
                self.cleanup_memory()
            
            # Process batch
            with self.profile_operation(f"batch_{i//batch_size}"):
                for tensor in batch:
                    mask = model.predict(tensor)
                    results.append(mask)
            
            # Cleanup after batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Batch processing complete: {len(results)} masks generated")
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report with metrics and statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        report = {
            'device': str(self.device),
            'gpu_enabled': self.metrics['gpu_enabled'],
            'cache_enabled': self.enable_cache,
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / 
                (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
                else 0.0
            ),
            'memory_warnings': self.metrics['memory_warnings'],
            'memory_usage': self.get_memory_usage()
        }
        
        if self.enable_profiling and self.metrics['profiling_data']:
            # Aggregate profiling data
            operations = {}
            for entry in self.metrics['profiling_data']:
                op_name = entry['operation']
                if op_name not in operations:
                    operations[op_name] = []
                operations[op_name].append(entry['duration'])
            
            report['profiling'] = {
                op_name: {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations)
                }
                for op_name, durations in operations.items()
            }
        
        return report


class ProfileContext:
    """Context manager for profiling operations."""
    
    def __init__(self, optimizer: PerformanceOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        if self.optimizer.enable_profiling:
            self.start_time = time.time()
            self.start_memory = self.optimizer.get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.optimizer.enable_profiling and self.start_time is not None:
            duration = time.time() - self.start_time
            end_memory = self.optimizer.get_memory_usage()
            
            profile_entry = {
                'operation': self.operation_name,
                'duration': duration,
                'start_memory': self.start_memory,
                'end_memory': end_memory,
                'memory_delta': {
                    'ram': end_memory['ram_used'] - self.start_memory['ram_used']
                }
            }
            
            if 'gpu_used' in end_memory and 'gpu_used' in self.start_memory:
                profile_entry['memory_delta']['gpu'] = (
                    end_memory['gpu_used'] - self.start_memory['gpu_used']
                )
            
            self.optimizer.metrics['profiling_data'].append(profile_entry)
            
            logger.debug(
                f"Profile: {self.operation_name} took {duration:.3f}s, "
                f"RAM delta: {profile_entry['memory_delta']['ram']:.1f} MB"
            )
