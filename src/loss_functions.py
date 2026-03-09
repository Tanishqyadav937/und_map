"""
Loss functions for road segmentation model training.

Implements Binary Cross-Entropy, Dice, and Focal loss functions
for semantic segmentation tasks. Includes a factory function to
create loss functions based on configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross-Entropy loss for binary segmentation.
    
    Computes the binary cross-entropy between predicted probabilities
    and ground truth binary masks. Applies sigmoid activation to logits
    before computing the loss.
    
    Formula: BCE = -[y * log(p) + (1-y) * log(1-p)]
    where y is ground truth and p is predicted probability.
    """
    
    def __init__(self):
        """Initialize Binary Cross-Entropy loss."""
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Cross-Entropy loss.
        
        Args:
            predictions: Model output logits with shape (B, 1, H, W)
            targets: Ground truth binary masks with shape (B, 1, H, W) or (B, H, W)
        
        Returns:
            Scalar loss value
        
        Preconditions:
            - predictions and targets have compatible shapes
            - targets contain only 0 and 1 values
            - predictions are raw logits (not probabilities)
        
        Postconditions:
            - Returns non-negative scalar tensor
            - Loss is differentiable for backpropagation
        """
        # Ensure targets have same shape as predictions
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Compute BCE loss with logits
        loss = self.bce_with_logits(predictions, targets)
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    
    Computes the Dice coefficient loss, which measures the overlap
    between predicted and ground truth masks. Particularly effective
    for imbalanced datasets where one class dominates.
    
    Formula: Dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    where X is prediction, Y is ground truth, and smooth prevents division by zero.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero (default: 1.0)
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Model output logits with shape (B, 1, H, W)
            targets: Ground truth binary masks with shape (B, 1, H, W) or (B, H, W)
        
        Returns:
            Scalar loss value
        
        Preconditions:
            - predictions and targets have compatible shapes
            - targets contain only 0 and 1 values
            - predictions are raw logits (not probabilities)
        
        Postconditions:
            - Returns non-negative scalar tensor in range [0, 1]
            - Loss is differentiable for backpropagation
            - Lower loss indicates better overlap
        """
        # Apply sigmoid to convert logits to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Ensure targets have same shape as predictions
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Compute Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss is 1 - Dice coefficient
        dice_loss = 1.0 - dice_coeff
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for binary segmentation.
    
    Addresses class imbalance by down-weighting easy examples and focusing
    on hard examples. Particularly useful when there's a large imbalance
    between foreground (road) and background (non-road) pixels.
    
    Formula: FL = -α * (1-p)^γ * log(p)
    where p is predicted probability, α is class weight, and γ is focusing parameter.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        
        Preconditions:
            - alpha is in range [0, 1]
            - gamma >= 0
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            predictions: Model output logits with shape (B, 1, H, W)
            targets: Ground truth binary masks with shape (B, 1, H, W) or (B, H, W)
        
        Returns:
            Scalar loss value
        
        Preconditions:
            - predictions and targets have compatible shapes
            - targets contain only 0 and 1 values
            - predictions are raw logits (not probabilities)
        
        Postconditions:
            - Returns non-negative scalar tensor
            - Loss is differentiable for backpropagation
            - Hard examples contribute more to loss than easy examples
        """
        # Apply sigmoid to convert logits to probabilities
        predictions_prob = torch.sigmoid(predictions)
        
        # Ensure targets have same shape as predictions
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy(predictions_prob, targets, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = predictions_prob * targets + (1 - predictions_prob) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Return mean loss
        return focal_loss.mean()


def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_type: Type of loss function ("bce", "dice", or "focal")
        **kwargs: Additional arguments for specific loss functions
            - smooth: Smoothing factor for Dice loss (default: 1.0)
            - alpha: Alpha parameter for Focal loss (default: 0.25)
            - gamma: Gamma parameter for Focal loss (default: 2.0)
    
    Returns:
        Loss function module (nn.Module)
    
    Raises:
        ValueError: If loss_type is not supported
    
    Preconditions:
        - loss_type is one of: "bce", "dice", "focal"
        - kwargs contain valid parameters for the specified loss type
    
    Postconditions:
        - Returns initialized loss function module
        - Loss function is ready for training
    
    Validates: Requirements 3.4
    
    Example:
        >>> bce_loss = create_loss_function("bce")
        >>> dice_loss = create_loss_function("dice", smooth=1.0)
        >>> focal_loss = create_loss_function("focal", alpha=0.25, gamma=2.0)
    """
    loss_type = loss_type.lower()
    
    if loss_type == "bce":
        logger.info("Creating Binary Cross-Entropy loss")
        return BinaryCrossEntropyLoss()
    
    elif loss_type == "dice":
        smooth = kwargs.get("smooth", 1.0)
        logger.info(f"Creating Dice loss with smooth={smooth}")
        return DiceLoss(smooth=smooth)
    
    elif loss_type == "focal":
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        logger.info(f"Creating Focal loss with alpha={alpha}, gamma={gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    else:
        error_msg = (
            f"Unsupported loss type: {loss_type}. "
            f"Supported types: 'bce', 'dice', 'focal'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


# Convenience function for getting loss function from config
def get_loss_function_from_config(config) -> nn.Module:
    """
    Get loss function from configuration object.
    
    Args:
        config: Configuration object with LOSS_FUNCTION attribute
    
    Returns:
        Loss function module
    
    Example:
        >>> from src.config import Config
        >>> loss_fn = get_loss_function_from_config(Config)
    """
    loss_type = config.LOSS_FUNCTION
    
    # Get additional parameters if available
    kwargs = {}
    if hasattr(config, 'DICE_SMOOTH'):
        kwargs['smooth'] = config.DICE_SMOOTH
    if hasattr(config, 'FOCAL_ALPHA'):
        kwargs['alpha'] = config.FOCAL_ALPHA
    if hasattr(config, 'FOCAL_GAMMA'):
        kwargs['gamma'] = config.FOCAL_GAMMA
    
    return create_loss_function(loss_type, **kwargs)
