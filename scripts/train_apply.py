import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import train_model
from test import test_model
from evaluate import evaluate_predictions, log_metrics
from utils.helper_functions import load_config  

def train_apply():
    """
    Wrapper function to train, test, and evaluate the model.
    """

    config_path = "config.yaml"
    
    # Load the configuration
    config = load_config(config_path)

    # Ensure checkpoint and logging directories exist
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training Phase
    print("Starting training phase...")
    train_model(config)

    # Testing Phase
    print("Starting testing phase...")
    predictions = test_model(config)

    # Evaluation Phase
    print("Starting evaluation phase...")
    results = evaluate_predictions(predictions, num_classes=config['data']['num_classes'])
    log_metrics(results, output_file=os.path.join(checkpoint_dir, "evaluation_results.txt"))
    print("Evaluation completed and metrics saved.")

    print("Train-Apply pipeline completed.")

if __name__ == "__main__":
    train_apply()
