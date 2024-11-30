import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import UNetGenerator 
from data.dataset import CustomDataset
from utils.helper_functions import load_config

def test_model(config):
    """
    Test the trained model on the test dataset.
    Args:
        config (dict): Configuration dictionary loaded from the config file.
    Returns:
        list: List of tuples (real_A, real_B, fake_B) for evaluation.
              - real_A: Input image.
              - real_B: Ground truth image.
              - fake_B: Generated image.
    """
    # Load the generator model
    generator = UNetGenerator()
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], "generator_latest.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()

    # Move the model to the correct device
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    generator.to(device)

    # Prepare the test dataset and dataloader
    test_dataset = CustomDataset(
        images_dir=config['data']['test_images_dir'],
        labels_dir=config['data']['test_labels_dir'],
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run predictions and store results
    predictions = []
    print("Running model on test dataset...")
    with torch.no_grad():
        for data in test_loader:
            real_A = data['A'].to(device)  # Input image
            real_B = data['B'].to(device)  # Ground truth image

            # Generate the fake image
            fake_B = generator(real_A)

            # Append the results for evaluation
            predictions.append((real_A.cpu(), real_B.cpu(), fake_B.cpu()))

    print(f"Testing complete. Processed {len(predictions)} samples.")
    return predictions

if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Test the model
    predictions = test_model(config)

    # Save predictions or perform any other processing as needed
    print(f"Generated {len(predictions)} predictions.")
