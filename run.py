import os
import cv2
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

train_path = './chest_Xray/train'
test_path = './chest_Xray/test'
valid_path = './chest_Xray/val'


def load_normal(norm_path):
    get_files = os.listdir(norm_path)
    normal_files = np.array(os.listdir(norm_path))
    
    
    norm_files = np.array(get_files)
    print('HERE\n',normal_files)
    norm_labels = np.array(['NORMAL']*len(norm_files))
    
    norm_images  = []
    for image in tqdm(norm_files):
        
        #percent by which the image is resized
        scale_percent = 50
        image = cv2.imread(norm_path+image)
        print("IMG: \n", image)
        # calculate the 50 percent of original dimension  
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        
        # dsize
        dsize = (width, height)
        image = cv2.resize(image, dsize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        norm_images.append(image) 
        
    norm_images = np.array(norm_images)
    return norm_images, norm_labels

def load_pneumonia(pneumo_path):
    pneumo_files = np.array(os.listdir(pneumo_path))
    pneumo_labels = np.array([pneumo_file.split('_')[1] for pneumo_file in pneumo_files])
    
    pneumo_images = [] 
    
    for image in tqdm(pneumo_files):
        try:
            
            image = cv2.imread(pneumo_path+image)
            image = cv2.resize(image, dsize=(200, 200))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pneumo_images.append(image) 
        except Exception as e:
            print('ERROR\n : ',e)
        
    pneumo_images = np.array(pneumo_images)
    
    return pneumo_images, pneumo_labels




train_dir =  './chest_Xray/train'

healthy_train_dir_= os.path.join(train_dir, 'NORMAL')
sick_train_dir = os.path.join(train_dir, 'PNEUMONIA')

norm_images, norm_labels = load_normal(healthy_train_dir_)
# pneumo_images, pneumo_labels = load_pneumonia('./chest_Xray/train/PNEUMONIA/')