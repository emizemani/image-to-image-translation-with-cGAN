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
            with open(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(file)), 'r') as f, open(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(file)), 'w') as g:
                g.write('<root>{}</root>'.format(f.read()))
            os.remove(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(file)))
    else:
        print(f"All files contain the necessary root structure <root> ... </root> ")

add_root("root")  


#Parse XML file, create list of dataframes, one dataframe contains all annotations for one picture

files_in_directory = os.listdir('./data/CMP_facade_DB_base/base/')
xml_files = [f for f in files_in_directory if f.endswith('.xml')]
object_annotation_list =[]

for filenumber, filename in enumerate(xml_files):
    # Parse the XML data
    root = ET.parse(os.path.join('./data/CMP_facade_DB_base/base/{}'.format(xml_files[filenumber])))

    # List to collect data for each object
    data_list = []

    # Iterate over each <object> element
    for obj in root.findall('object'):
        # Extract x and y values
        x_values = [float(x.text) for x in obj.find('points').findall('x')]
        y_values = [float(y.text) for y in obj.find('points').findall('y')]
        label = int(obj.find('label').text)
        labelname = obj.find('labelname').text
        flag = int(obj.find('flag').text)

        # Construct a dictionary for this object
        data = {
            "x1": x_values[0],
            "x2": x_values[1],
            "y1": y_values[0],
            "y2": y_values[1],
            "label": label,
            "labelname": labelname,
            "flag": flag
        }
        
        # Append the dictionary to the list
        data_list.append(data)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    print ("df:")
    print(df)
    df.info(verbose=True)
    print(df.head())

object_annotation_list.append(df)

print (object_annotation_list[0])

#next: change object_annotation_list to dictionary with filename as key

#check
print('check')
