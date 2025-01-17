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
    
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    generator.load_state_dict(torch.load(checkpoint_path, device, weights_only=True))
    generator.eval()

    # Move the model to the correct device
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

def save_test_samples(real_A, real_B, fake_B, number, save_dir):
    """
    Save test sample images.
    
    Args:
        real_A (Tensor): Input images
        real_B (Tensor): Ground truth images
        fake_B (Tensor): Generated images
        number (int): Current image number
        save_dir (str): Directory to save the images
    """
    import torchvision.utils as vutils

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Concatenate the images horizontally
    comparison = torch.cat([real_A, real_B, fake_B], dim=3)
    
    # Save the image
    vutils.save_image(
        comparison,
        os.path.join(save_dir, f'image_{number}.png'),
        normalize=True
    )

if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define storage directory
    save_dir = "validation/test1"

    # Test the model
    predictions = test_model(config)

    # Save predictions or perform any other processing as needed
    for i, prediction in enumerate(predictions):
        save_test_samples(*prediction, i+1, save_dir)
    print(f"Generated {len(predictions)} images.")
