import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        Args:
            images_dir (str): Directory with the input images (label maps).
            labels_dir (str): Directory with the target images (realistic facades).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.transform = transform if transform else transforms.ToTensor()  # Default to tensor transformation

    def __len__(self): ### Overloading the len() operator, lets you make instances of your classes behave like familiar Python objects, even though they are custom data types
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the label map and realistic image for the given index
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        # Open images
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Return a dictionary with both image and label
        return {"A": image, "B": label}


