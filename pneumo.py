
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import zlib
import itertools
import sklearn
import itertools
import scipy
import skimage
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from keras.applications.mobilenet import MobileNet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# %matplotlib inline


import warnings
warnings.filterwarnings("ignore")


train_dir = './chest_Xray/train/'
test_dir = './chest_Xray/test/'
val_dir = './chest_Xray/val/'

def get_data(folder):
    X = []
    y = []
    data_gen = ImageDataGenerator()
    test_train = data_gen.flow_from_directory(train_dir, class_mode='binary')
    
    batchX, batchy = test_train.next()
    print('TEST\n',(batchX.shape, batchX.min(), batchX.max()))
    
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                print('NORMAL OK')
                label = 0
            elif folderName in ['PNEUMONIA']:
                print('Pneumo OK')

                label = 1
            else:
                print('ELSE ICI')
                label = 2
            for img_file_name in tqdm(folder+ folderName):
                print(img_file_name)
                break
                img_file = cv2.imread(folder+folderName+'/'+ img_file_name)
                
                if img_file is not None:
                    print('HERE 4')
                    break
                    # img_file = skimage.transform.resize(img_file,(150,150,3))
                    
                    # img_arr = np.asarray(img_file)
                    # X.append(img_arr)
                    # y.append(label)
    # X = np.asarray(X)
    # y = np.asarray(y)
    # return X,y

X_train, y_train = get_data(train_dir)

#encore labels to hot vectors (2-> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes=2)
# y_testHot = to_categorical(y_test, num_classes)