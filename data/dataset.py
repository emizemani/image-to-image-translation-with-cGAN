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

    def __len__(self):
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


### self parameter
'''
class Dog:
    def __init__(self, name):
        self.name = name  # Using 'self' to refer to the instance's attribute

    def bark(self):
        print(f"{self.name} says woof!")

# Create an instance of Dog
my_dog = Dog("Buddy")
my_dog.bark()  # Output: Buddy says woof!


Explanation
__init__(self, name): The __init__ method is called when an instance of the class is created. 
The self.name = name line uses self to set an attribute name on the instance. 
Each Dog instance will have its own name attribute, distinct from other instances.
Calling bark(): When my_dog.bark() is called, Python translates this to Dog.bark(my_dog). 
The instance my_dog is automatically passed as self in the bark method, so self.name refers to my_dog.name, which is "Buddy".

short:
self is a reference to the instance on which a method is being called.
It allows methods to access or modify the instances own attributes and other methods.
self must always be the first parameter in instance methods, although you dont pass it explicitly when calling the method — Python handles it automatically.

'''