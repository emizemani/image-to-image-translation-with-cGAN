import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import train_model
from test import test_model
from evaluate import evaluate_predictions, log_metrics, extract_best_median_worst
from utils.helper_functions import load_config


def plot_generalization_error(metrics, output_dir):
    """
    Plot generalization error metrics as violin plots.
    Args:
        metrics (dict): Dictionary of evaluation metrics.
        output_dir (str): Directory to save the plots.
    """
    data = []
    for metric, values in metrics.items():
        if isinstance(values, list):  # Only consider per-sample metrics
            for value in values:
                data.append({"Metric": metric, "Value": value})

    sns.violinplot(data=data, x="Metric", y="Value")
    plt.title("Generalization Error Metrics")
    plt.savefig(os.path.join(output_dir, "generalization_error.png"))
    plt.close()


def train_apply():
    """
    Train, test, and evaluate the model with hyperparameter tuning.
    """
    # Load the configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Ensure checkpoint and logging directories exist
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_model_path = None
    best_metric = float('-inf')  # Track the best SSIM score
    learning_rates = [0.0001, 0.0002]
    batch_sizes = [16, 32]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTesting config: LR={lr}, Batch Size={batch_size}")
            config['training']['lr'] = lr
            config['training']['batch_size'] = batch_size

            # Train and Test
            train_model(config)
            predictions = test_model(config)

            # Evaluate
            results = evaluate_predictions(predictions, num_classes=config['data']['num_classes'])
            log_metrics(results, output_file=os.path.join(checkpoint_dir, f"evaluation_results_lr{lr}_bs{batch_size}.txt"))

            # Save the best model based on multiple metrics (e.g., SSIM, MSE)
            current_metric = results["SSIM"] - results["MSE"]  # Example composite metric
            if current_metric > best_metric:
                best_metric = current_metric
                best_model_path = os.path.join(checkpoint_dir, f"generator_lr{lr}_bs{batch_size}.pth")
                torch.save(generator.state_dict(), best_model_path)
                print(f"New best model saved: {best_model_path}")

    print(f"Best model achieved with Metric (Composite Score): {best_metric}")

    # Final Evaluation Phase with the Best Model
    if best_model_path is not None:
        print("Loading the best model for evaluation...")
        generator = UNetGenerator()
        generator.load_state_dict(torch.load(best_model_path))
        generator.eval()

        # Test and Evaluate the Best Model
        print("Generating predictions with the best model...")
        predictions = test_model(config)

        print("Evaluating the best model's predictions...")
        results = evaluate_predictions(predictions, num_classes=config['data']['num_classes'])
        log_metrics(results, output_file=os.path.join(checkpoint_dir, "best_model_evaluation_results.txt"))

        # Plot Generalization Error
        print("Plotting generalization error metrics for the best model...")
        plot_generalization_error(results, checkpoint_dir)

        # Showcase Test Cases
        print("Extracting best, median, and worst test cases...")
        examples = extract_best_median_worst(predictions, results, metric="SSIM")
        for case, (real_B, fake_B, score) in examples.items():
            real_B.save(os.path.join(checkpoint_dir, f"{case}_real.png"))
            fake_B.save(os.path.join(checkpoint_dir, f"{case}_fake.png"))
            print(f"{case.capitalize()} case saved with score: {score}")
    else:
        print("No valid model was found during hyperparameter tuning.")

    print("Train-Apply pipeline completed.")


if __name__ == "__main__":
    train_apply()
