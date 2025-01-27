import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.helper_functions import load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import PatchGANDiscriminator
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from torchvision import transforms

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
    real_loader = DataLoader(real_dataset, batch_size=len(real_dataset), shuffle=False)

    fake_dataset = CustomDataset(
        images_dir=f"{input_dir}/2",
        labels_dir=f"{input_dir}/0",
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
        is_training=False
    )
    fake_loader = DataLoader(fake_dataset, batch_size=len(fake_dataset), shuffle=False)

    # Run predictions and store results
    with torch.no_grad():
        for data in real_loader:
            real_A = data['A'].to(device)  # Input image
            real_B = data['B'].to(device)  # Ground truth image

            real_features = discriminator(real_B, real_A, return_features=True)

    with torch.no_grad():
        for data in fake_loader:
            real_A = data['A'].to(device)  # Input image
            fake_B = data['B'].to(device)  # fake image

            fake_features = discriminator(fake_B, real_A, return_features=True)

    print('1')

    # Flatten the feature maps for both real and fake images
    real_features_flat = real_features.view(real_features.size(0), -1).detach().numpy()
    fake_features_flat = fake_features.view(fake_features.size(0), -1).detach().numpy()

    # Concatenate the real and fake feature maps into one array
    all_features = np.concatenate([real_features_flat, fake_features_flat], axis=0)

    # Create labels for visualization (1 for real, 0 for fake)
    labels = np.concatenate([np.ones(real_features_flat.shape[0]), np.zeros(fake_features_flat.shape[0])])

    # Apply t-SNE to reduce the dimensionality of the feature maps
    tsne = TSNE(n_components=2)  # Reduce to 2D
    tsne_results = tsne.fit_transform(all_features)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='coolwarm')
    plt.title("t-SNE Visualization: Real vs Fake Images")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Add legend to indicate real vs fake
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    
    plt.savefig("t-sne_visualization.pdf")



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
