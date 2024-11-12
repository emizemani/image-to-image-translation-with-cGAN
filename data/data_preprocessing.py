import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode: The interactive mode is mainly useful if you build plots from the command line and want to see the effect of each command while you are building the figure.

#https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder.data
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/felix/Desktop/PML/PML code/CMP_facade_DB_base/base/cmp_b0001.xml')
root = tree.getroot()

for child in root:
    print(child.tag, child.attrib)

print(root[0][0].text)
print(root[0][1].text)


