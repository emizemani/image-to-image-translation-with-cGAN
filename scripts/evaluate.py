from utils.metrics import pixel_accuracy, per_class_accuracy, iou, structural_similarity, mean_squared_error

def evaluate_predictions(predictions, num_classes=3):
    """
    Evaluate predictions using various metrics.
    Args:
        predictions (list): List of tuples (real_A, real_B, fake_B).
                           - real_A: Input image (optional, not used here but could be).
                           - real_B: Ground truth image/tensor.
                           - fake_B: Predicted/generated image/tensor.
        num_classes (int): Number of classes for classification/segmentation tasks.
    Returns:
        dict: Evaluation metrics as a dictionary.
    """
    # Initialize accumulators for batch-wise metrics
    total_pixel_accuracy = 0
    total_ssim = 0
    total_mse = 0
    total_iou = {cls: 0 for cls in range(num_classes)}
    total_per_class_acc = {cls: 0 for cls in range(num_classes)}

    num_samples = len(predictions)

    for real_B, fake_B in predictions:
        # Ensure tensors are on the CPU for evaluation
        real_B = real_B.cpu()
        fake_B = fake_B.cpu()

        # Calculate individual metrics
        total_pixel_accuracy += pixel_accuracy(fake_B, real_B)
        total_ssim += structural_similarity(fake_B, real_B)
        total_mse += mean_squared_error(fake_B, real_B)

        # IoU and per-class accuracy (only relevant for multi-class segmentation)
        iou_scores = iou(fake_B, real_B, num_classes=num_classes)
        per_class_acc = per_class_accuracy(fake_B, real_B, num_classes=num_classes)

        for cls in range(num_classes):
            total_iou[cls] += iou_scores[cls]
            total_per_class_acc[cls] += per_class_acc[cls]

    # Normalize metrics by the number of samples
    avg_pixel_accuracy = total_pixel_accuracy / num_samples
    avg_ssim = total_ssim / num_samples
    avg_mse = total_mse / num_samples
    avg_iou = {cls: total_iou[cls] / num_samples for cls in total_iou}
    avg_per_class_acc = {cls: total_per_class_acc[cls] / num_samples for cls in total_per_class_acc}

    # Compile results into a dictionary
    results = {
        "Pixel Accuracy": avg_pixel_accuracy,
        "SSIM": avg_ssim,
        "MSE": avg_mse,
        "IoU": avg_iou,
        "Per-Class Accuracy": avg_per_class_acc,
    }

    return results

def log_metrics(results, output_file="evaluation_results.txt"):
    """
    Log evaluation metrics to a file.
    Args:
        results (dict): Evaluation metrics dictionary.
        output_file (str): File to save the metrics.
    """
    with open(output_file, "w") as file:
        file.write("Evaluation Metrics:\n")
        for metric, value in results.items():
            if isinstance(value, dict):  # For IoU and Per-Class Accuracy
                file.write(f"{metric}:\n")
                for cls, score in value.items():
                    file.write(f"  Class {cls}: {score:.4f}\n")
            else:
                file.write(f"{metric}: {value:.4f}\n")
    print(f"Metrics logged to {output_file}")
