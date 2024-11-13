import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode: The interactive mode is mainly useful if you build plots from the command line and want to see the effect of each command while you are building the figure.



#preprocessing should follow this structure as mentioned in the instructions: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html