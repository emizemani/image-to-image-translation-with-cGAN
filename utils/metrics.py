import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy.
    Args:
        pred (torch.Tensor): Predicted tensor (binary or multi-class, shape: BxCxHxW).
        target (torch.Tensor): Ground truth tensor (binary or multi-class, shape: BxCxHxW).
    Returns:
        float: Pixel-wise accuracy.
    """
    pred = pred.argmax(dim=1) if pred.ndim == 4 else pred
    target = target.argmax(dim=1) if target.ndim == 4 else target
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def structural_similarity(pred, target):
    """
    Calculate Structural Similarity Index Measure (SSIM).
    Args:
        pred (torch.Tensor): Predicted tensor (shape: Bx3xHxW).
        target (torch.Tensor): Ground truth tensor (shape: Bx3xHxW).
    Returns:
        float: Mean SSIM for the batch.
    """
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC
    target = target.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC
    batch_size = pred.shape[0]
    ssim_scores = []
    for i in range(batch_size):
        ssim_score = ssim(
            pred[i], 
            target[i], 
            data_range=target[i].max() - target[i].min(),
            channel_axis=-1,
            win_size=3  # Using smallest possible window size
        )
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)


def mean_squared_error(pred, target):
    """
    Calculate Mean Squared Error (MSE).
    Args:
        pred (torch.Tensor): Predicted tensor (shape: Bx3xHxW).
        target (torch.Tensor): Ground truth tensor (shape: Bx3xHxW).
    Returns:
        float: Mean squared error.
    """
    mse = torch.mean((pred - target) ** 2).item()
    return mse
