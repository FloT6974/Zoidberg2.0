import os, os.path, shutil
from os import path
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
# %matplotlib inline
import random 
import numpy as np
import pandas as pd
import seaborn as sns 
from PIL import Image

# Machine Learning
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.preprocessing import LabelEncoder

#Deep Learning
import tensorflow as tf
import keras
from keras.utils import plot_model 
from keras import backend as K 
from keras import metrics
from keras.regularizers import l2,l1

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer, Activation
from keras.preprocessing.image import  ImageDataGenerator
from keras.metrics import AUC
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pydot
from dask import bag,  diagnostics
from mlxtend.plotting import plot_confusion_matrix



class bcolors:
    HEADER = '\033[95m' # purple
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m' # end color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    


# Show the image path
Dataset_dir ='./chest_Xray/'

train_dir =Dataset_dir+'/train/'
test_dir =Dataset_dir+'/test/'
val_dir = Dataset_dir+'/val/'

train_NORMAL_dir = train_dir+'/NORMAL/'
train_PNEUMONIA_dir = train_dir+'/PNEUMONIA/'
new_dir = 'split/'

# Load and copy data
def load_data():
    print(f"{bcolors.OKCYAN} Load data ...{bcolors.ENDC}")
    

    images_NORMAL = [file for file in os.listdir(train_NORMAL_dir) if file.endswith('.jpeg')]
    images_PNEUMONIA = [file for file in os.listdir(train_PNEUMONIA_dir) if file.endswith('.jpeg')]
   
    print(f"{bcolors.HEADER}There are, {len(images_NORMAL)}, NORMAL images{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}There are, {len(images_PNEUMONIA)}, PNEUMONIA images{bcolors.ENDC}")

    #Plot number of classes to identify imbalances
    number_classes = {'NORMAL':1583,
                    'PNEUMONIA':4273}
    # plt.figure(figsize=(16, 9))
    plt.bar(number_classes.keys(), number_classes.values(), width = 0.5)
    plt.title("Number of images by Class")
    plt.xlabel("Class Name")
    plt.ylabel("Numer of Images")
    plt.show()
    # plt.style.use("ggplot")
    # plt.figure(figsize=(16,9))
    plt.pie(x=number_classes.values(), labels=number_classes.keys(), autopct='%1.1f')
    plt.title("PIE Image category Distrib")
    plt.show()
    # Redo Train-Val-Test Split
    # Creat folders and subfolders to get a hierarchical file structure
    # Create a new folder 'split'
    checker = False
    if(path.exists('split')):
        print(f"{bcolors.WARNING}ALREADY EXISTE SPLIT FOLDER{bcolors.ENDC}\n")

        checker = True 
        
    else:
        os.mkdir(new_dir)
        checker = False
        print(f'{bcolors.OKGREEN}split folder created\n{bcolors.ENDC}\n')



    # create subfolder train under split

    train_folder = os.path.join(new_dir, 'train')
    # Create subfolders 'train_NORMAL' and 'train_PNEUMONIA' under the 'train'
    train_NORMAL = os.path.join(train_folder, 'NORMAL')
    train_PNEUMONIA = os.path.join(train_folder, 'PNEUMONIA')

    #----------------------------------------------------------------
    # Create a subfolder 'test' under the 'split'
    test_folder = os.path.join(new_dir, 'test')
    # Create subfolders 'test_NORMAL' and 'test_PNEUMONIA' under the 'test'
    test_NORMAL = os.path.join(test_folder, 'NORMAL')
    test_PNEUMONIA = os.path.join(test_folder, 'PNEUMONIA')
    #----------------------------------------------------------------

    # Create a subfolder 'test' under the 'split'
    val_folder = os.path.join(new_dir, 'val')
    # # Create subfolders 'val_NORMAL' and 'val_PNEUMONIA' under the 'test'
    val_NORMAL = os.path.join(val_folder, 'NORMAL')
    val_PNEUMONIA = os.path.join(val_folder, 'PNEUMONIA')

    # Use all the path strings to make new directories
    if checker == False:
        os.mkdir(train_folder)
        os.mkdir(train_NORMAL)
        os.mkdir(train_PNEUMONIA)

        os.mkdir(test_folder)
        os.mkdir(test_NORMAL)
        os.mkdir(test_PNEUMONIA)

        os.mkdir(val_folder)
        os.mkdir(val_NORMAL)
        os.mkdir(val_PNEUMONIA)
        print(f'{bcolors.OKGREEN} train_folder => NORMAL/, PNEUMONIA/ \n test_folder => NORMAL/, PNEUMONIA/ \n val_folder => NORMAL/, PNEUMONIA/ \n created !!{bcolors.ENDC}\n')

    else:
        print(f'{bcolors.WARNING}ALREADY EXISTS FOLDERS:\n train_folder => NORMAL/, PNEUMONIA/ \n test_folder => NORMAL/, PNEUMONIA/ \n val_folder => NORMAL/, PNEUMONIA/{bcolors.ENDC} \n')



    # Use a 70%/20%/10% split for train/validation/test
    print(f'{bcolors.HEADER}Number of images to train')
    print('# train_NORMAL: 20% ', round(len(images_NORMAL)*0.2))
    print('# train_NORMAL: 100%', len(images_NORMAL))

    print('# train_PNEUMONIA: ', round(len(images_PNEUMONIA)*0.2))
    print('________________________________________________')
    print('Number of images to validation')
    print('# val_NORMAL: ', round(len(images_NORMAL)*0.2))
    print('# val_PNEUMONIA: ', round(len(images_PNEUMONIA)*0.2))
    print('________________________________________________')
    print('Number of images to test')
    print('# test_NORMAL: ', round(len(images_NORMAL)*0.1))
    print('# test_PNEUMONIA: ', round(len(images_PNEUMONIA)*0.1))
    print(f'\n{bcolors.ENDC}',)
    
    # Copy files and dir
    # train NORMAL
    imgs = images_NORMAL[:1108]
    for img in imgs:
        origin = os.path.join(train_NORMAL_dir, img)
        dest = os.path.join(train_NORMAL,img)
        shutil.copyfile(origin,dest)
    # validation NORMAL
    imgs = images_NORMAL[1108:1425]
    for img in imgs:
        origin = os.path.join(train_NORMAL_dir, img)
        destination = os.path.join(val_NORMAL, img)
        shutil.copyfile(origin, destination)
    # test NORMAL
    imgs = images_NORMAL[1425:]
    for img in imgs:
        origin = os.path.join(train_NORMAL_dir, img)
        destination = os.path.join(test_NORMAL, img)
        shutil.copyfile(origin, destination)
        
    
    # train PNEUMONIA
    imgs = images_PNEUMONIA
    for img in imgs[:2991]:
        origin = os.path.join(train_PNEUMONIA_dir, img)
        destination = os.path.join(train_PNEUMONIA, img)
        shutil.copyfile(origin, destination)
    # validation PNEUMONIA
    imgs = images_PNEUMONIA[2991:3846]
    for img in imgs:
        origin = os.path.join(train_PNEUMONIA_dir, img)
        destination = os.path.join(val_PNEUMONIA, img)
        shutil.copyfile(origin, destination)
    # test PNEUMONIA
    imgs = images_PNEUMONIA[3846:]
    for img in imgs:
        origin = os.path.join(train_PNEUMONIA_dir, img)
        destination = os.path.join(test_PNEUMONIA, img)
        shutil.copyfile(origin, destination)
        
    return train_dir, val_dir, test_dir

  
  
  
  
  
def model_test():
    classifier = Sequential()
    
    classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(activation='relu', units=128))
    classifier.add(Dense(activation='sigmoid', units=1))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    
    

    train_data_gen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)


    training_set = train_data_gen.flow_from_directory(
        './chest_Xray/train',
        target_size=(64,64),
        batch_size=32,
        class_mode ='binary'
    )

    test_set = test_datagen.flow_from_directory(
        './chest_Xray/test',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary'
    )

    history = classifier.fit(training_set, 
                             steps_per_epoch=len(training_set),
                             epochs=3,
                             validation_data=test_set,
                             validation_steps=len(test_set),
                             )
    
    #Accuracy
    print(history.history.keys())
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Training set', 'Test set'], loc='upper left')
    # plt.show()
    
    #Loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.title('Loss')
    # plt.ylabel('Loss/Epoch')
    # plt.xlabel('Epoch')
    # plt.legend(['Training set', 'Test set'], loc='upper left')
    # plt.show()
    
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(h.history["loss"], label='train_loss')
    # plt.plot(h.history["val_loss"], label='val_loss')
    # plt.plot(h.history["accuracy"], label='train_accuracy')
    # plt.plot(h.history["val_accuracy"], label='val_accuracy')
    plt.title('MODEL TRAINING')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend() 
    

# load_data()
model_test()
plt.show()
