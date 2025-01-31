import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from utils.helper_functions import load_config
from src.model import PatchGANDiscriminator
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from torchvision import transforms

def analyze_discriminator(config, input_dir, output_dir):

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
    discriminator.eval()

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
    real_tsne = real_features.view(real_features.size(0), -1).detach().cpu().numpy()
    fake_tsne = fake_features.view(fake_features.size(0), -1).detach().cpu().numpy()

    # Concatenate the real and fake feature maps into one array
    all_features = np.concatenate([real_tsne, fake_tsne], axis=0)

    # Apply t-SNE to reduce the dimensionality of the feature maps
    tsne = TSNE(n_components=2)  # Reduce to 2D
    tsne_results = tsne.fit_transform(all_features)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='blue', label='Real Images', marker='o')
    plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], c='red', label='Fake Images', marker='x')
    plt.title("t-SNE Visualization: Real & Fake Images")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Add legend to indicate real vs fake
    plt.legend()
    
    plt.savefig("t-sne_visualization_new3.pdf")


    # # Fit a GMM to the 2D t-SNE results
    # gmm = GaussianMixture(n_components=2, random_state=42)
    # gmm.fit(tsne_results)

    # # Predict cluster assignments
    # clusters = gmm.predict(tsne_results)

    # # Optional: Analyze cluster separation
    # num_real = real_features_flat.shape[0]  # Number of real samples
    # real_clusters = clusters[:num_real]    # Clusters assigned to real images
    # fake_clusters = clusters[num_real:]    # Clusters assigned to fake images

    # # Visualize t-SNE results with cluster assignments
    # plt.figure(figsize=(8, 6))

    # # Plot real images with clusters
    # plt.scatter(tsne_results[:num_real, 0], tsne_results[:num_real, 1], 
    #             c=clusters[:num_real], cmap='viridis', marker='o', label='Real', alpha=0.7)

    # # Plot fake images with clusters
    # plt.scatter(tsne_results[num_real:, 0], tsne_results[num_real:, 1], 
    #             c=clusters[num_real:], cmap='viridis', marker='x', label='Fake', alpha=0.7)

    # plt.title("t-SNE Visualization with GMM Clusters")
    # plt.legend()
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.colorbar(label="Cluster")

    # plt.savefig("t-sne_visualization_with_clusters.pdf")


if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define input directory
    input_dir = "validation/test_prototyp21"

    # Define output directory
    output_dir = "explain/test2"

    analyze_discriminator(config, input_dir, output_dir)
