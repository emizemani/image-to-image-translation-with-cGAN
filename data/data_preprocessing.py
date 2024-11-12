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

def add_root(search_filename):
    '''Add a root structure <root> ... </root> to the XML-file to be able to use the XML parser
       The dataset called "Base dataset" from https://cmp.felk.cvut.cz/~tylecr1/facade/ is originally missing the root 
    '''
    # Get the list of all files in the current directory
    files_in_directory = os.listdir('./data/CMP_facade_DB_base/base/')
    
    # Filter only .xml files 
    xml_files = [f for f in files_in_directory if f.endswith('.xml')]

    # Check if the search_filename is already present in any of the .xml filenames
    initial_files = [f for f in xml_files if search_filename not in f]
    
    if initial_files:
        for file in initial_files:
            #Add a root to the XML-file to be able to use the XML parser
            with open(os.path.join('./data/CMP_facade_DB_base/base//{}'.format(file)), 'r') as f, open(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(file)), 'w') as g:
                g.write('<root>{}</root>'.format(f.read()))
            os.remove(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(file)))
    else:
        print(f"All files contain the necessary root structure <root> ... </root> ")

add_root("root")  



#https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.TreeBuilder.data
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/felix/Desktop/PML/PML code/CMP_facade_DB_base/base/cmp_b0001root.xml')
root = tree.getroot()


for child in root.findall('object'):
    print(child.tag, child.attrib)

print(root[0][0].text)
print(root[0][1].text)
print ("XXXXXXXXXXXXXXXX")
