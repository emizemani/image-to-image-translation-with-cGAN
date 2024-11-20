import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import CustomDataset  # Ensure this path is correct
import yaml
from scripts import train





def show_samples(pair_number=1):

    config = train.load_config()

    train_dataset = CustomDataset(
        images_dir=config['data']['train_images_dir'],
        labels_dir=config['data']['train_labels_dir'])
    
    return 0








if __name__ == "__show_samples__":
    show_samples()

print("done")