import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import CustomDataset 
from scripts.train import load_config
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show(image_label_pair=None, outfile=None):

    config = load_config()

    train_dataset = CustomDataset(
        images_dir=config['data']['train_images_dir'],
        labels_dir=config['data']['train_labels_dir'])
    #Debug
    #print(len(train_dataset))
    
    batch=[]
    for pair in image_label_pair:
        batch.append(train_dataset[pair]["A"])
        batch.append(train_dataset[pair]["B"])

    # Create a grid of images
    grid = make_grid(batch, nrow=2) 

    # Convert from (C, H, W) to (H, W, C) and display
    grid_image = grid.permute(1, 2, 0)
    plt.imshow(grid_image)
    plt.axis("off")
    plt.text(110,-10,"Image")
    plt.text(350,-10,"Label")

    if outfile==None:
        plt.show()
    else:
        plt.savefig('images_to_show.png')
        
    return 

# Show pairs of images and associated labels
images_to_show = [1,2,3,10,23]
show(images_to_show)
