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

def per_class_accuracy(pred, target, num_classes):
    """
    Calculate accuracy per class.
    Args:
        pred (torch.Tensor): Predicted tensor (shape: BxHxW or BxCxHxW).
        target (torch.Tensor): Ground truth tensor (shape: BxHxW or BxCxHxW).
        num_classes (int): Number of classes.
    Returns:
        dict: Per-class accuracy as a dictionary.
    """
    pred = pred.argmax(dim=1) if pred.ndim == 4 else pred
    target = target.argmax(dim=1) if target.ndim == 4 else target
    class_acc = {}
    for cls in range(num_classes):
        cls_mask = (target == cls)
        correct = (pred[cls_mask] == cls).sum().item()
        total = cls_mask.sum().item()
        class_acc[cls] = correct / total if total > 0 else 0.0
    return class_acc

def iou(pred, target, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class.
    Args:
        pred (torch.Tensor): Predicted tensor (shape: BxHxW or BxCxHxW).
        target (torch.Tensor): Ground truth tensor (shape: BxHxW or BxCxHxW).
        num_classes (int): Number of classes.
    Returns:
        dict: IoU for each class as a dictionary.
    """
    pred = pred.argmax(dim=1) if pred.ndim == 4 else pred
    target = target.argmax(dim=1) if target.ndim == 4 else target
    iou_scores = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        iou_scores[cls] = intersection / union if union > 0 else 0.0
    return iou_scores

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
            pred[i], target[i], data_range=target[i].max() - target[i].min(), multichannel=True
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
