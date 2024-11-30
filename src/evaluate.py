from utils.metrics import pixel_accuracy, structural_similarity, mean_squared_error
import numpy as np
from PIL import Image


def evaluate_predictions(predictions):
    """
    Evaluate predictions using various metrics.
    Args:
        predictions (list): List of tuples (real_B, fake_B).
                            - real_B: Ground truth image/tensor.
                            - fake_B: Predicted/generated image/tensor.
    Returns:
        dict: Evaluation metrics as a dictionary.
    """
    # Initialize accumulators for batch-wise metrics
    total_pixel_accuracy = 0
    total_ssim = []
    total_mse = []

    num_samples = len(predictions)

    for real_B, fake_B in predictions:
        # Ensure tensors are on the CPU for evaluation
        real_B = real_B.cpu()
        fake_B = fake_B.cpu()

        # Calculate individual metrics
        total_pixel_accuracy += pixel_accuracy(fake_B, real_B)
        total_ssim.append(structural_similarity(fake_B, real_B))
        total_mse.append(mean_squared_error(fake_B, real_B))

    # Normalize metrics by the number of samples
    avg_pixel_accuracy = total_pixel_accuracy / num_samples
    avg_ssim = np.mean(total_ssim)
    avg_mse = np.mean(total_mse)

    # Compile results into a dictionary
    results = {
        "Pixel Accuracy": avg_pixel_accuracy,
        "SSIM": avg_ssim,
        "MSE": avg_mse,
        "Sample SSIM Scores": total_ssim,  # For best/median/worst cases
    }

    return results


def extract_best_median_worst(predictions, results, metric="SSIM"):
    """
    Extract the best, median, and worst test cases based on a given metric.
    Args:
        predictions (list): List of tuples (real_B, fake_B).
                            - real_B: Ground truth image.
                            - fake_B: Generated image.
        results (dict): Dictionary of evaluation results containing per-sample metrics.
        metric (str): The metric to rank the predictions (default: "SSIM").
    Returns:
        dict: Dictionary with best, median, and worst examples.
    """
    scores = results.get("Sample SSIM Scores", [])
    if not scores:
        raise ValueError(f"Metric '{metric}' scores not found in results.")

    # Get indices of best, median, and worst scores
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    median_idx = np.argsort(scores)[len(scores) // 2]

    cases = {
        "best": predictions[best_idx],
        "median": predictions[median_idx],
        "worst": predictions[worst_idx],
    }

    return {
        case: (Image.fromarray((real_B.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)),  # Convert to PIL Image
               Image.fromarray((fake_B.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)),
               scores[idx])
        for case, (real_B, fake_B, idx) in zip(cases.keys(), [(best_idx, best_idx, best_idx), (median_idx, median_idx, median_idx), (worst_idx, worst_idx, worst_idx)])
    }


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
            if isinstance(value, list):  # For per-sample metrics
                file.write(f"{metric}: List of {len(value)} values (not displayed here).\n")
            else:
                file.write(f"{metric}: {value:.4f}\n")
    print(f"Metrics logged to {output_file}")
