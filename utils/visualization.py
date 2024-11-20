import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import CustomDataset 
from scripts.train import load_config
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_samples(image_label_pair_1=1, image_label_pair_2=2):

    config = load_config()

    train_dataset = CustomDataset(
        images_dir=config['data']['train_images_dir'],
        labels_dir=config['data']['train_labels_dir'])
    
    batch = [train_dataset[image_label_pair_1]["A"], train_dataset[image_label_pair_1]["B"], train_dataset[image_label_pair_2]["A"], train_dataset[image_label_pair_2]["B"]]

    # Create a grid of images
    grid = make_grid(batch, nrow=2) 

    # Convert from (C, H, W) to (H, W, C) and display
    grid_image = grid.permute(1, 2, 0)
    plt.imshow(grid_image)
    plt.axis("off")
    plt.text(110,-10,"Image")
    plt.text(350,-10,"Label")
    plt.show()

    print(len(train_dataset))
    
    return 

# Show two pairs of images and associated labels
show_samples(5,3)
