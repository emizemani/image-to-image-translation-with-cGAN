from utils.metrics import structural_similarity, mean_squared_error
import numpy as np
from PIL import Image

def evaluate_predictions(predictions):
    """
    Evaluate predictions using SSIM and MSE.
    Args:
        predictions (list): List of tuples (real_B, fake_B).
                            - real_B: Ground truth image/tensor.
                            - fake_B: Predicted/generated image/tensor.
    Returns:
        dict: Evaluation metrics as a dictionary.
    """
    total_ssim = []
    total_mse = []

    for real_B, fake_B in predictions:
        total_ssim.append(structural_similarity(fake_B, real_B))
        total_mse.append(mean_squared_error(fake_B, real_B))

    avg_ssim = np.mean(total_ssim)
    avg_mse = np.mean(total_mse)

    return {
        "SSIM": avg_ssim,
        "MSE": avg_mse,
        "Sample SSIM Scores": total_ssim,  # For extracting best/median/worst cases
    }

def extract_best_median_worst(predictions, metrics):
    """
    Extract the best, median, and worst test cases based on SSIM.
    Args:
        predictions (list): List of tuples (real_B, fake_B).
                            - real_B: Ground truth image.
                            - fake_B: Generated image.
        metrics (dict): Dictionary of evaluation results containing per-sample SSIM scores.
    Returns:
        dict: Dictionary with best, median, and worst examples.
    """
    scores = metrics.get("Sample SSIM Scores", [])
    if not scores:
        raise ValueError("No SSIM scores found in metrics.")

    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    median_idx = np.argsort(scores)[len(scores) // 2]

    cases = {
        "best": predictions[best_idx],
        "median": predictions[median_idx],
        "worst": predictions[worst_idx],
    }

    return {
        case: (predictions[idx][0], predictions[idx][1], scores[idx])
        for case, idx in zip(cases.keys(), [best_idx, median_idx, worst_idx])
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
