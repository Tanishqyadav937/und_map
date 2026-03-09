"""
RoadSegmentationModel component for semantic segmentation of satellite images.
Identifies road pixels using U-Net or DeepLabV3+ architectures.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from typing import Optional, Dict, Any, List
import logging
import os

from src.config import Config
from src.morphological_processor import MorphologicalProcessor

logger = logging.getLogger(__name__)


class RoadSegmentationModel:
    """
    Semantic segmentation model for road detection in satellite imagery.
    
    Supports U-Net and DeepLabV3+ architectures with pretrained ImageNet encoders.
    Handles device detection (CUDA/CPU) and model placement automatically.
    
    Attributes:
        model: The underlying segmentation model (U-Net or DeepLabV3+)
        device: The device (cuda or cpu) where the model is placed
        architecture: The model architecture name ("unet" or "deeplabv3plus")
        encoder_name: The encoder backbone name (e.g., "resnet34")
    """
    
    def __init__(
        self,
        architecture: str = Config.MODEL_ARCHITECTURE,
        encoder_name: str = Config.ENCODER_NAME,
        encoder_weights: Optional[str] = Config.ENCODER_WEIGHTS,
        in_channels: int = Config.IN_CHANNELS,
        out_classes: int = Config.OUT_CLASSES,
        device: Optional[str] = None
    ):
        """
        Initialize the RoadSegmentationModel with specified architecture.
        
        Args:
            architecture: Model architecture ("unet" or "deeplabv3plus")
            encoder_name: Encoder backbone (e.g., "resnet34", "resnet50", "efficientnet-b0")
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels (3 for RGB)
            out_classes: Number of output classes (1 for binary segmentation)
            device: Device to place model on ("cuda", "cpu", or None for auto-detect)
        
        Raises:
            ValueError: If architecture is not supported
            RuntimeError: If model initialization fails
        
        Preconditions:
            - architecture must be "unet" or "deeplabv3plus"
            - in_channels must be positive integer
            - out_classes must be positive integer
            - encoder_name must be supported by segmentation-models-pytorch
        
        Postconditions:
            - self.model is initialized with specified architecture
            - self.device is set to available device (CUDA if available, else CPU)
            - Model is placed on the correct device
            - Model is in evaluation mode by default
        """
        self.architecture = architecture.lower()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_classes = out_classes
        
        # Detect and set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing {self.architecture} model with {encoder_name} encoder")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model based on architecture
        if self.architecture == "unet":
            try:
                self.model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=out_classes
                )
            except Exception as e:
                logger.error(f"Failed to initialize U-Net model: {e}")
                raise RuntimeError(f"U-Net initialization failed: {e}")
        elif self.architecture == "deeplabv3plus":
            try:
                self.model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=out_classes
                )
            except Exception as e:
                logger.error(f"Failed to initialize DeepLabV3Plus model: {e}")
                raise RuntimeError(f"DeepLabV3Plus initialization failed: {e}")
        else:
            error_msg = (
                f"Unsupported architecture: {architecture}. "
                f"Supported architectures: 'unet', 'deeplabv3plus'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode by default
        self.model.eval()
        
        # Initialize morphological processor
        self.morphological_processor = MorphologicalProcessor(
            apply_closing=Config.APPLY_MORPHOLOGICAL_CLOSING,
            kernel_size=Config.MORPHOLOGICAL_KERNEL_SIZE,
            iterations=Config.MORPHOLOGICAL_ITERATIONS
        )
        
        logger.info(f"Model initialized successfully on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary containing model architecture, encoder, device, and parameter count
        """
        return {
            "architecture": self.architecture,
            "encoder_name": self.encoder_name,
            "encoder_weights": self.encoder_weights,
            "in_channels": self.in_channels,
            "out_classes": self.out_classes,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters())
        }
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save model weights and configuration to a checkpoint file.
        
        Saves the model's state dictionary along with configuration metadata
        for reproducibility. The checkpoint includes model architecture, encoder
        settings, and training state.
        
        Args:
            checkpoint_path: Path where checkpoint file will be saved
        
        Raises:
            IOError: If checkpoint cannot be written to disk
            RuntimeError: If model state cannot be serialized
        
        Preconditions:
            - checkpoint_path is valid file path
            - Directory for checkpoint_path exists or can be created
            - Model is initialized with valid state
        
        Postconditions:
            - Checkpoint file is created at checkpoint_path
            - Checkpoint contains model weights and configuration
            - Checkpoint can be loaded with load_checkpoint()
        
        Validates: Requirements 13.1, 13.5
        """
        import os
        
        try:
            # Create directory if it doesn't exist
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Prepare checkpoint dictionary with model state and config
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'architecture': self.architecture,
                'encoder_name': self.encoder_name,
                'encoder_weights': self.encoder_weights,
                'in_channels': self.in_channels,
                'out_classes': self.out_classes,
                'model_info': self.get_model_info()
            }
            
            # Save checkpoint to disk
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"Checkpoint saved successfully to {checkpoint_path}")
            logger.debug(f"Checkpoint contains {len(checkpoint['model_state_dict'])} state dict entries")
            
        except IOError as e:
            error_msg = f"Failed to write checkpoint to {checkpoint_path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)
        except Exception as e:
            error_msg = f"Failed to save checkpoint: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint file with validation.
        
        Loads model weights and validates that the checkpoint is compatible
        with the current model architecture. Performs integrity checks to
        detect corrupted or incompatible checkpoints.
        
        Args:
            checkpoint_path: Path to checkpoint file to load
        
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint is incompatible with model architecture
            RuntimeError: If checkpoint is corrupted or cannot be loaded
        
        Preconditions:
            - checkpoint_path points to valid checkpoint file
            - Checkpoint was created by save_checkpoint()
            - Model is initialized with compatible architecture
        
        Postconditions:
            - Model weights are loaded from checkpoint
            - Model architecture matches checkpoint configuration
            - Model is ready for inference or training
        
        Validates: Requirements 13.2, 13.3, 13.4
        """
        import os
        
        # Validate checkpoint file exists
        if not os.path.exists(checkpoint_path):
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load checkpoint from disk
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'architecture', 'encoder_name', 
                           'in_channels', 'out_classes']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                error_msg = (
                    f"Checkpoint is corrupted or invalid. Missing keys: {missing_keys}. "
                    f"Expected keys: {required_keys}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Validate architecture compatibility
            if checkpoint['architecture'] != self.architecture:
                error_msg = (
                    f"Architecture mismatch: checkpoint has '{checkpoint['architecture']}', "
                    f"but model is '{self.architecture}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if checkpoint['encoder_name'] != self.encoder_name:
                error_msg = (
                    f"Encoder mismatch: checkpoint has '{checkpoint['encoder_name']}', "
                    f"but model is '{self.encoder_name}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if checkpoint['in_channels'] != self.in_channels:
                error_msg = (
                    f"Input channels mismatch: checkpoint has {checkpoint['in_channels']}, "
                    f"but model has {self.in_channels}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if checkpoint['out_classes'] != self.out_classes:
                error_msg = (
                    f"Output classes mismatch: checkpoint has {checkpoint['out_classes']}, "
                    f"but model has {self.out_classes}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Load model state dictionary
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            logger.debug(f"Loaded {len(checkpoint['model_state_dict'])} state dict entries")
            logger.debug(f"Model architecture: {checkpoint['architecture']}, "
                        f"encoder: {checkpoint['encoder_name']}")
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except RuntimeError as e:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # Catch any other exceptions and wrap as RuntimeError
            error_msg = f"Failed to load checkpoint from {checkpoint_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    def save_checkpoint(
        self, 
        checkpoint_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Save model weights and configuration to a checkpoint file.

        Saves the model's state dictionary along with configuration metadata
        for reproducibility. The checkpoint includes model architecture, encoder
        settings, hyperparameters, and random seed.

        Args:
            checkpoint_path: Path where checkpoint file will be saved
            hyperparameters: Optional dictionary of hyperparameters for reproducibility
            random_seed: Optional random seed used for training

        Raises:
            IOError: If checkpoint cannot be written to disk
            RuntimeError: If model state cannot be serialized

        Preconditions:
            - checkpoint_path is valid file path
            - Directory for checkpoint_path exists or can be created
            - Model is initialized with valid state

        Postconditions:
            - Checkpoint file is created at checkpoint_path
            - Checkpoint contains model weights, configuration, and hyperparameters
            - Checkpoint can be loaded with load_checkpoint()

        Validates: Requirements 13.1, 13.5, 20.3
        """
        import os

        try:
            # Create directory if it doesn't exist
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            # Prepare checkpoint dictionary with model state and config
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'architecture': self.architecture,
                'encoder_name': self.encoder_name,
                'encoder_weights': self.encoder_weights,
                'in_channels': self.in_channels,
                'out_classes': self.out_classes,
                'model_info': self.get_model_info()
            }
            
            # Add hyperparameters if provided
            if hyperparameters is not None:
                checkpoint['hyperparameters'] = hyperparameters
                logger.debug(f"Saved {len(hyperparameters)} hyperparameters")
            
            # Add random seed if provided
            if random_seed is not None:
                checkpoint['random_seed'] = random_seed
                logger.debug(f"Saved random seed: {random_seed}")

            # Save checkpoint to disk
            torch.save(checkpoint, checkpoint_path)

            logger.info(f"Checkpoint saved successfully to {checkpoint_path}")
            logger.debug(f"Checkpoint contains {len(checkpoint['model_state_dict'])} state dict entries")

        except IOError as e:
            error_msg = f"Failed to write checkpoint to {checkpoint_path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)
        except Exception as e:
            error_msg = f"Failed to save checkpoint: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint file with validation.

        Loads model weights and validates that the checkpoint is compatible
        with the current model architecture. Performs integrity checks to
        detect corrupted or incompatible checkpoints.

        Args:
            checkpoint_path: Path to checkpoint file to load

        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint is incompatible with model architecture
            RuntimeError: If checkpoint is corrupted or cannot be loaded

        Preconditions:
            - checkpoint_path points to valid checkpoint file
            - Checkpoint was created by save_checkpoint()
            - Model is initialized with compatible architecture

        Postconditions:
            - Model weights are loaded from checkpoint
            - Model architecture matches checkpoint configuration
            - Model is ready for inference or training

        Validates: Requirements 13.2, 13.3, 13.4
        """
        import os

        # Validate checkpoint file exists
        if not os.path.exists(checkpoint_path):
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load checkpoint from disk
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'architecture', 'encoder_name',
                           'in_channels', 'out_classes']
            missing_keys = [key for key in required_keys if key not in checkpoint]

            if missing_keys:
                error_msg = (
                    f"Checkpoint is corrupted or invalid. Missing keys: {missing_keys}. "
                    f"Expected keys: {required_keys}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Validate architecture compatibility
            if checkpoint['architecture'] != self.architecture:
                error_msg = (
                    f"Architecture mismatch: checkpoint has '{checkpoint['architecture']}', "
                    f"but model is '{self.architecture}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if checkpoint['encoder_name'] != self.encoder_name:
                error_msg = (
                    f"Encoder mismatch: checkpoint has '{checkpoint['encoder_name']}', "
                    f"but model is '{self.encoder_name}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if checkpoint['in_channels'] != self.in_channels:
                error_msg = (
                    f"Input channels mismatch: checkpoint has {checkpoint['in_channels']}, "
                    f"but model has {self.in_channels}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if checkpoint['out_classes'] != self.out_classes:
                error_msg = (
                    f"Output classes mismatch: checkpoint has {checkpoint['out_classes']}, "
                    f"but model has {self.out_classes}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Load model state dictionary
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Set model to evaluation mode
            self.model.eval()

            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            logger.debug(f"Loaded {len(checkpoint['model_state_dict'])} state dict entries")
            logger.debug(f"Model architecture: {checkpoint['architecture']}, "
                        f"encoder: {checkpoint['encoder_name']}")

        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except RuntimeError as e:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # Catch any other exceptions and wrap as RuntimeError
            error_msg = f"Failed to load checkpoint from {checkpoint_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def predict(self, image_tensor: torch.Tensor, threshold: float = 0.5):
        """
        Generate binary road mask from satellite image tensor.
        
        Performs inference on a single preprocessed image tensor and returns
        a binary road mask where 1 indicates road pixels and 0 indicates non-road.
        
        Args:
            image_tensor: Preprocessed image tensor with shape (3, H, W) or (1, 3, H, W)
            threshold: Threshold for binary classification (default: 0.5)
        
        Returns:
            Binary numpy array with shape (H, W) containing only 0 and 1 values
        
        Raises:
            ValueError: If image_tensor has invalid shape or dimensions
            RuntimeError: If inference fails
        
        Preconditions:
            - image_tensor has shape (3, H, W) or (1, 3, H, W)
            - image_tensor values are normalized (typically in [0, 1] range)
            - threshold is in range (0, 1)
            - Model is in evaluation mode
        
        Postconditions:
            - Returns binary numpy array with shape (H, W)
            - Array contains only 0 (non-road) and 1 (road) values
            - Output dimensions match input spatial dimensions
            - No gradient computation (inference only)
        
        Validates: Requirements 2.1, 2.2, 2.3
        """
        # Validate threshold
        if not 0 < threshold < 1:
            raise ValueError(f"Threshold must be in range (0, 1), got {threshold}")
        
        # Validate input tensor shape
        if image_tensor.dim() == 3:
            # Shape is (C, H, W), add batch dimension
            if image_tensor.shape[0] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} channels, got {image_tensor.shape[0]}"
                )
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
        elif image_tensor.dim() == 4:
            # Shape is (B, C, H, W)
            if image_tensor.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size of 1, got {image_tensor.shape[0]}. "
                    f"Use predict() for single images only."
                )
            if image_tensor.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} channels, got {image_tensor.shape[1]}"
                )
        else:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions, got {image_tensor.dim()} dimensions"
            )
        
        # Store original spatial dimensions
        _, _, height, width = image_tensor.shape
        
        # Move tensor to model device
        image_tensor = image_tensor.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform inference without gradient computation
        try:
            with torch.no_grad():
                # Forward pass through model
                logits = self.model(image_tensor)  # Shape: (1, 1, H, W)
                
                # Apply sigmoid activation to get probabilities
                probabilities = torch.sigmoid(logits)  # Shape: (1, 1, H, W)
                
                # Apply threshold to generate binary mask
                binary_mask = (probabilities > threshold).float()  # Shape: (1, 1, H, W)
                
                # Remove batch and channel dimensions
                binary_mask = binary_mask.squeeze(0).squeeze(0)  # Shape: (H, W)
                
                # Move to CPU and convert to numpy
                binary_mask_np = binary_mask.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")
        
        # Validate output shape matches input spatial dimensions
        if binary_mask_np.shape != (height, width):
            raise RuntimeError(
                f"Output shape {binary_mask_np.shape} does not match "
                f"input spatial dimensions ({height}, {width})"
            )
        
        # Validate output is binary (only 0 and 1)
        unique_values = np.unique(binary_mask_np)
        if not np.all(np.isin(unique_values, [0.0, 1.0])):
            raise RuntimeError(
                f"Output contains non-binary values: {unique_values}"
            )
        
        # Convert to integer type for cleaner output
        binary_mask_np = binary_mask_np.astype(np.uint8)
        
        # Apply morphological post-processing if enabled
        if Config.APPLY_MORPHOLOGICAL_CLOSING:
            logger.debug("Applying morphological post-processing")
            binary_mask_np = self.morphological_processor.process(binary_mask_np)
        
        logger.debug(
            f"Prediction complete: shape={binary_mask_np.shape}, "
            f"road_pixels={np.sum(binary_mask_np)}, "
            f"road_ratio={np.mean(binary_mask_np):.3f}"
        )
        
        return binary_mask_np
    
    def predict_batch(
        self,
        image_tensors: List[torch.Tensor],
        threshold: float = 0.5,
        batch_size: int = 4
    ) -> List[np.ndarray]:
        """
        Generate binary road masks for multiple images using batch processing.
        
        Processes multiple images in batches to improve GPU utilization and throughput.
        This is more efficient than calling predict() multiple times for GPU inference.
        
        Args:
            image_tensors: List of preprocessed image tensors, each with shape (3, H, W)
            threshold: Threshold for binary classification (default: 0.5)
            batch_size: Number of images to process per batch (default: 4)
        
        Returns:
            List of binary numpy arrays, each with shape (H, W)
        
        Raises:
            ValueError: If image_tensors is empty or contains invalid tensors
            RuntimeError: If batch inference fails
        
        Preconditions:
            - image_tensors is non-empty list
            - All tensors have shape (3, H, W)
            - All tensors have same spatial dimensions
            - threshold is in range (0, 1)
            - batch_size > 0
        
        Postconditions:
            - Returns list of binary masks with same length as input
            - Each mask contains only 0 and 1 values
            - Output dimensions match input spatial dimensions
        
        Validates: Requirement 11.4 (batch processing)
        """
        if not image_tensors:
            raise ValueError("image_tensors cannot be empty")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if not 0 < threshold < 1:
            raise ValueError(f"Threshold must be in range (0, 1), got {threshold}")
        
        logger.info(f"Batch prediction: {len(image_tensors)} images, batch_size={batch_size}")
        
        # Validate all tensors have same shape
        first_shape = image_tensors[0].shape
        for i, tensor in enumerate(image_tensors):
            if tensor.dim() != 3:
                raise ValueError(
                    f"Tensor {i} has {tensor.dim()} dimensions, expected 3 (C, H, W)"
                )
            if tensor.shape != first_shape:
                raise ValueError(
                    f"Tensor {i} has shape {tensor.shape}, expected {first_shape}"
                )
        
        # Set model to evaluation mode
        self.model.eval()
        
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(image_tensors), batch_size):
                batch_tensors = image_tensors[i:i + batch_size]
                
                # Stack tensors into batch
                batch = torch.stack(batch_tensors).to(self.device)  # Shape: (B, C, H, W)
                
                # Perform inference without gradient computation
                with torch.no_grad():
                    # Forward pass through model
                    logits = self.model(batch)  # Shape: (B, 1, H, W)
                    
                    # Apply sigmoid activation
                    probabilities = torch.sigmoid(logits)
                    
                    # Apply threshold
                    binary_masks = (probabilities > threshold).float()
                    
                    # Remove channel dimension and convert to numpy
                    binary_masks = binary_masks.squeeze(1)  # Shape: (B, H, W)
                    binary_masks_np = binary_masks.cpu().numpy().astype(np.uint8)
                
                # Add individual masks to results
                for mask in binary_masks_np:
                    results.append(mask)
                
                logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_tensors)} images")
        
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise RuntimeError(f"Batch inference failed: {e}")
        
        logger.info(f"Batch prediction complete: {len(results)} masks generated")
        
        return results

    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        loss_function,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        checkpoint_dir: str = "models",
        checkpoint_name: str = "best_model.pth",
        random_seed: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train the segmentation model with epoch iteration and validation.

        Implements the complete training loop including:
        - Forward and backward passes
        - Optimizer updates (Adam, SGD, AdamW)
        - Validation after each epoch
        - Best model checkpointing based on validation loss
        - Training and validation metrics logging
        - Hyperparameter and random seed saving for reproducibility

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            loss_function: Loss function module (nn.Module)
            optimizer_type: Optimizer type ("adam", "sgd", or "adamw")
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            checkpoint_name: Name of checkpoint file for best model
            random_seed: Random seed for reproducibility (optional)

        Returns:
            Dictionary containing training history:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch
                - 'best_epoch': Epoch number with best validation loss
                - 'best_val_loss': Best validation loss achieved

        Raises:
            ValueError: If optimizer_type is not supported
            RuntimeError: If training fails

        Preconditions:
            - train_loader and val_loader are valid DataLoaders
            - loss_function is a valid nn.Module
            - num_epochs > 0
            - learning_rate > 0
            - optimizer_type in ["adam", "sgd", "adamw"]

        Postconditions:
            - Model is trained for num_epochs
            - Best model checkpoint is saved to disk with hyperparameters
            - Training history is returned
            - Model is in evaluation mode

        Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6, 20.3
        """
        import os
        from tqdm import tqdm

        # Validate inputs
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            logger.info(f"Using Adam optimizer with lr={learning_rate}")
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            logger.info(f"Using SGD optimizer with lr={learning_rate}, momentum=0.9")
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            logger.info(f"Using AdamW optimizer with lr={learning_rate}")
        else:
            error_msg = (
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported types: 'adam', 'sgd', 'adamw'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Prepare hyperparameters dictionary for saving
        hyperparameters = {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'optimizer_type': optimizer_type,
            'loss_function': str(loss_function.__class__.__name__),
            'batch_size': train_loader.batch_size,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'architecture': self.architecture,
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'in_channels': self.in_channels,
            'out_classes': self.out_classes
        }
        
        if random_seed is not None:
            hyperparameters['random_seed'] = random_seed
            logger.info(f"Training with random seed: {random_seed}")

        # Initialize training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Batch size: {train_loader.batch_size}")

        try:
            # Training loop
            for epoch in range(1, num_epochs + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch}/{num_epochs}")
                logger.info(f"{'='*60}")

                # Training phase
                self.model.train()
                train_loss = 0.0
                train_batches = 0

                # Progress bar for training
                train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

                for batch_idx, (images, masks) in enumerate(train_pbar):
                    # Move data to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = self.model(images)

                    # Compute loss
                    loss = loss_function(predictions, masks)

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    # Accumulate loss
                    train_loss += loss.item()
                    train_batches += 1

                    # Update progress bar
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Calculate average training loss
                avg_train_loss = train_loss / train_batches
                train_losses.append(avg_train_loss)

                logger.info(f"Training Loss: {avg_train_loss:.4f}")

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_batches = 0

                # Progress bar for validation
                val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")

                with torch.no_grad():
                    for batch_idx, (images, masks) in enumerate(val_pbar):
                        # Move data to device
                        images = images.to(self.device)
                        masks = masks.to(self.device)

                        # Forward pass
                        predictions = self.model(images)

                        # Compute loss
                        loss = loss_function(predictions, masks)

                        # Accumulate loss
                        val_loss += loss.item()
                        val_batches += 1

                        # Update progress bar
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # Calculate average validation loss
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)

                logger.info(f"Validation Loss: {avg_val_loss:.4f}")

                # Save best model checkpoint with hyperparameters
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch

                    # Save checkpoint with hyperparameters and random seed
                    self.save_checkpoint(
                        checkpoint_path,
                        hyperparameters=hyperparameters,
                        random_seed=random_seed
                    )

                    logger.info(f"✓ New best model saved! Validation loss: {best_val_loss:.4f}")
                else:
                    logger.info(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")

            # Training complete
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Complete!")
            logger.info(f"{'='*60}")
            logger.info(f"Best epoch: {best_epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Best model saved to: {checkpoint_path}")

            # Set model to evaluation mode
            self.model.eval()

            # Return training history
            return {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }

        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"RoadSegmentationModel("
            f"architecture={self.architecture}, "
            f"encoder={self.encoder_name}, "
            f"device={self.device})"
        )
    
    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        loss_function,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        checkpoint_dir: str = "models",
        checkpoint_name: str = "best_model.pth"
    ) -> Dict[str, list]:
        """
        Train the segmentation model with epoch iteration and validation.
        
        Implements the complete training loop including:
        - Forward and backward passes
        - Optimizer updates (Adam, SGD, AdamW)
        - Validation after each epoch
        - Best model checkpointing based on validation loss
        - Training and validation metrics logging
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            loss_function: Loss function module (nn.Module)
            optimizer_type: Optimizer type ("adam", "sgd", or "adamw")
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            checkpoint_name: Name of checkpoint file for best model
            
        Returns:
            Dictionary containing training history:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch
                - 'best_epoch': Epoch number with best validation loss
                - 'best_val_loss': Best validation loss achieved
        
        Raises:
            ValueError: If optimizer_type is not supported
            RuntimeError: If training fails
            
        Preconditions:
            - train_loader and val_loader are valid DataLoaders
            - loss_function is a valid nn.Module
            - num_epochs > 0
            - learning_rate > 0
            - optimizer_type in ["adam", "sgd", "adamw"]
            
        Postconditions:
            - Model is trained for num_epochs
            - Best model checkpoint is saved to disk
            - Training history is returned
            - Model is in evaluation mode
            
        Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6
        """
        from tqdm import tqdm
        
        # Validate inputs
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        
        # Create optimizer based on type
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            logger.info(f"Using Adam optimizer with lr={learning_rate}")
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            logger.info(f"Using SGD optimizer with lr={learning_rate}, momentum=0.9")
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            logger.info(f"Using AdamW optimizer with lr={learning_rate}")
        else:
            error_msg = (
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported types: 'adam', 'sgd', 'adamw'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Initialize training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        
        try:
            # Training loop
            for epoch in range(1, num_epochs + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch}/{num_epochs}")
                logger.info(f"{'='*60}")
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_batches = 0
                
                # Progress bar for training
                train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
                
                for batch_idx, (images, masks) in enumerate(train_pbar):
                    # Move data to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Compute loss
                    loss = loss_function(predictions, masks)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Accumulate loss
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Update progress bar
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Calculate average training loss
                avg_train_loss = train_loss / train_batches
                train_losses.append(avg_train_loss)
                
                logger.info(f"Training Loss: {avg_train_loss:.4f}")
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                # Progress bar for validation
                val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
                
                with torch.no_grad():
                    for batch_idx, (images, masks) in enumerate(val_pbar):
                        # Move data to device
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        
                        # Forward pass
                        predictions = self.model(images)
                        
                        # Compute loss
                        loss = loss_function(predictions, masks)
                        
                        # Accumulate loss
                        val_loss += loss.item()
                        val_batches += 1
                        
                        # Update progress bar
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Calculate average validation loss
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                
                # Save best model checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    
                    # Save checkpoint
                    self.save_checkpoint(checkpoint_path)
                    
                    logger.info(f"✓ New best model saved! Validation loss: {best_val_loss:.4f}")
                else:
                    logger.info(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
            
            # Training complete
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Complete!")
            logger.info(f"{'='*60}")
            logger.info(f"Best epoch: {best_epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Best model saved to: {checkpoint_path}")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Return training history
            return {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def predict(self, image_tensor: torch.Tensor, threshold: float = 0.5) -> 'np.ndarray':
        """
        Generate binary road mask from satellite image tensor.

        Performs inference on a single preprocessed image tensor and returns
        a binary road mask where 1 indicates road pixels and 0 indicates non-road.

        Args:
            image_tensor: Preprocessed image tensor with shape (3, H, W) or (1, 3, H, W)
            threshold: Threshold for binary classification (default: 0.5)

        Returns:
            Binary numpy array with shape (H, W) containing only 0 and 1 values

        Raises:
            ValueError: If image_tensor has invalid shape or dimensions
            RuntimeError: If inference fails

        Preconditions:
            - image_tensor has shape (3, H, W) or (1, 3, H, W)
            - image_tensor values are normalized (typically in [0, 1] range)
            - threshold is in range (0, 1)
            - Model is in evaluation mode

        Postconditions:
            - Returns binary numpy array with shape (H, W)
            - Array contains only 0 (non-road) and 1 (road) values
            - Output dimensions match input spatial dimensions
            - No gradient computation (inference only)

        Validates: Requirements 2.1, 2.2, 2.3
        """
        import numpy as np

        # Validate threshold
        if not 0 < threshold < 1:
            raise ValueError(f"Threshold must be in range (0, 1), got {threshold}")

        # Validate input tensor shape
        if image_tensor.dim() == 3:
            # Shape is (C, H, W), add batch dimension
            if image_tensor.shape[0] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} channels, got {image_tensor.shape[0]}"
                )
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
        elif image_tensor.dim() == 4:
            # Shape is (B, C, H, W)
            if image_tensor.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size of 1, got {image_tensor.shape[0]}. "
                    f"Use predict() for single images only."
                )
            if image_tensor.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} channels, got {image_tensor.shape[1]}"
                )
        else:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions, got {image_tensor.dim()} dimensions"
            )

        # Store original spatial dimensions
        _, _, height, width = image_tensor.shape

        # Move tensor to model device
        image_tensor = image_tensor.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Perform inference without gradient computation
        try:
            with torch.no_grad():
                # Forward pass through model
                logits = self.model(image_tensor)  # Shape: (1, 1, H, W)

                # Apply sigmoid activation to get probabilities
                probabilities = torch.sigmoid(logits)  # Shape: (1, 1, H, W)

                # Apply threshold to generate binary mask
                binary_mask = (probabilities > threshold).float()  # Shape: (1, 1, H, W)

                # Remove batch and channel dimensions
                binary_mask = binary_mask.squeeze(0).squeeze(0)  # Shape: (H, W)

                # Move to CPU and convert to numpy
                binary_mask_np = binary_mask.cpu().numpy()

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")

        # Validate output shape matches input spatial dimensions
        if binary_mask_np.shape != (height, width):
            raise RuntimeError(
                f"Output shape {binary_mask_np.shape} does not match "
                f"input spatial dimensions ({height}, {width})"
            )

        # Validate output is binary (only 0 and 1)
        unique_values = np.unique(binary_mask_np)
        if not np.all(np.isin(unique_values, [0.0, 1.0])):
            raise RuntimeError(
                f"Output contains non-binary values: {unique_values}"
            )

        # Convert to integer type for cleaner output
        binary_mask_np = binary_mask_np.astype(np.uint8)

        logger.debug(
            f"Prediction complete: shape={binary_mask_np.shape}, "
            f"road_pixels={np.sum(binary_mask_np)}, "
            f"road_ratio={np.mean(binary_mask_np):.3f}"
        )

        return binary_mask_np
