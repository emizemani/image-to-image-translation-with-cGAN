import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import CustomDataset 
from scripts.train import load_config
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show(image_label_pair, dataset, outfile=None, title_col_left="Image", title_col_right="Label"):
    '''dataset need to be instance from class CustomDataset'''

    batch = []
    for pair in image_label_pair:
        batch.append(dataset[pair]["A"])
        batch.append(dataset[pair]["B"])
    # Create a grid of images
    grid = make_grid(batch, nrow=2) 

    # Convert from (C, H, W) to (H, W, C) and display
    grid_image = grid.permute(1, 2, 0)
    plt.imshow(grid_image)
    plt.axis("off")
    plt.text(grid_image.size(dim=1)*0.25,-25,title_col_left,ha='center',va='center')
    plt.text(grid_image.size(dim=1)*0.75,-25,title_col_right,ha='center',va='center')

    if outfile==None:
        plt.show()
    else:
        plt.savefig('images_to_show.png')

    return 

# Get images and associated labels from training dataset
config = load_config()
dataset_to_show = CustomDataset(
    images_dir=config['data']['train_images_dir'],
    labels_dir=config['data']['train_labels_dir'])

# Show pairs of images and associated labels
images_to_show = [1,3,10,23]
show(images_to_show, dataset_to_show)
