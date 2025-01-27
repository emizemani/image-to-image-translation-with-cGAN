import sys
import os
import torch
from utils.helper_functions import load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import PatchGANDiscriminator


def analyze_discriminator(config, input_dir, output_dir):
    # load inputs

    # load discriminator
    discriminator = PatchGANDiscriminator()
    
    # Use the same paths as in train_apply.py
    best_discriminator_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/discriminator_latest.pth')
    
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    # Load with appropriate device mapping
    discriminator.load_state_dict(
        torch.load(best_discriminator_path, device, weights_only=True)
    )
    discriminator = discriminator.to(device)

    # Prepare the test dataset and dataloader
    real_dataset = CustomDataset(
        images_dir=f"{input_dir}/1",
        labels_dir=f"{input_dir}/0",
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
        is_training=False
    )
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)

    fake_dataset = CustomDataset(
        images_dir=f"{input_dir}/2",
        labels_dir=f"{input_dir}/0",
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
        is_training=False
    )
    fake_loader = DataLoader(fake_dataset, batch_size=1, shuffle=False)

    # Run predictions and store results
    with torch.no_grad():
        for data in real_loader:
            real_A = data['A'].to(device)  # Input image
            real_B = data['B'].to(device)  # Ground truth image

            features = discriminator(real_B, real_A, return_features=True)

            print(features)

            # Append the results for evaluation
            # predictions.append((real_A.cpu(), real_B.cpu(), fake_B.cpu()))

    with torch.no_grad():
        for data in fake_dataset:
            real_A = data['A'].to(device)  # Input image
            fake_B = data['B'].to(device)  # Ground truth image

            # Generate the fake image
            features = generator(fake_B, real_A, return_features=True)

            print(features)

            # Append the results for evaluation
            # predictions.append((real_A.cpu(), real_B.cpu(), fake_B.cpu()))



    return








if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define input directory
    input_dir = "validation/test1"

    # Define output directory
    output_dir = "validation/test2"

    analyze_discriminator(config, input_dir, output_dir)
