import os, os.path, shutil
from os import path
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

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
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
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
Dataset_dir ='./chest_Xray'

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
    number_classes = {'NORMAL':len(images_NORMAL),
                    'PNEUMONIA':len(images_PNEUMONIA)}
    
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
    print(f'test divise img\n{bcolors.ENDC}',)
    # Copy files and dir
    # train NORMAL
    # imgs = images_NORMAL[:1108]
    
    # for img in imgs:
    #     origin = os.path.join(train_NORMAL_dir, img)
    #     dest = os.path.join(train_NORMAL,img)
    #     shutil.copyfile(origin,dest)
    # # validation NORMAL
    # imgs = images_NORMAL[1108:1425]
    # for img in imgs:
    #     origin = os.path.join(train_NORMAL_dir, img)
    #     destination = os.path.join(val_NORMAL, img)
    #     shutil.copyfile(origin, destination)
    # # test NORMAL
    # imgs = images_NORMAL[1425:]
    # for img in imgs:
    #     origin = os.path.join(train_NORMAL_dir, img)
    #     destination = os.path.join(test_NORMAL, img)
    #     shutil.copyfile(origin, destination)
        
    
    # train PNEUMONIA
    # imgs = images_PNEUMONIA
    # for img in imgs[:2991]:
    #     origin = os.path.join(train_PNEUMONIA_dir, img)
    #     destination = os.path.join(train_PNEUMONIA, img)
    #     shutil.copyfile(origin, destination)
    # # validation PNEUMONIA
    # imgs = images_PNEUMONIA[2991:3846]
    # for img in imgs:
    #     origin = os.path.join(train_PNEUMONIA_dir, img)
    #     destination = os.path.join(val_PNEUMONIA, img)
    #     shutil.copyfile(origin, destination)
    # # test PNEUMONIA
    # imgs = images_PNEUMONIA[3846:]
    # for img in imgs:
    #     origin = os.path.join(train_PNEUMONIA_dir, img)
    #     destination = os.path.join(test_PNEUMONIA, img)
    #     shutil.copyfile(origin, destination)
        
    return train_dir, val_dir, test_dir
        
        
    
    
    
def data_generator(train_datagen, train_dir,val_datagen, val_dir,test_datagen, test_dir):
    datagen_lst = [train_datagen, val_datagen, test_datagen]
    directory_lst = [train_dir,val_dir,test_dir]
    generator_lst = []
    
    for generator, directory in zip(datagen_lst, directory_lst):
        print('HERERfhsgfgqegfegfyegfqkffqfgkqfrkfqwfjgqwvfgfgwfgwehkfwhqgfhqwefweqkfwqfwf\n',generator)
        if directory == train_dir:
            shuffle = True
        else:
            shuffle = False
        g = generator.flow_from_directory(directory = directory,
                                         target_size = (64,64),
                                         batch_size = 128,
                                         color_mode = 'grayscale',
                                         class_mode = 'binary',
                                         shuffle = shuffle,
                                         seed = 1)
        generator_lst.append(g)
    
    return generator_lst

  # Load the images

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator,val_generator, test_generator = data_generator(
    train_datagen, train_dir,
    val_datagen, val_dir,
    test_datagen, test_dir)

def model_1():
    cnn = Sequential()
    
    cnn.add(InputLayer(input_shape=(64,64,1)))
    # Convolution
    cnn.add(Conv2D(32, (3, 3), activation='relu',input_shape=(64,64,3)))
    
    # Pooling
    cnn.add(MaxPooling2D((2, 2))) 
    cnn.add(BatchNormalization(axis=1))
    
    #  Convolution 2
    cnn.add(Conv2D(32, (3, 3),activation='relu'))
    #  Pooling 2
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(BatchNormalization(axis=1))
    
    # cnn.add(Conv2D(32, (3, 3), activation='relu',padding='same'))  
    # cnn.add(MaxPooling2D((2, 2),padding='same'))
    # cnn.add(BatchNormalization(axis=1))
    
    
    cnn.add(Flatten()) #convert 3D features map in 1D
    
    # cnn.add(Dropout(0.5))
    # cnn.add(Dense(64))
    cnn.add(Dense(activation='relu',units =128))
    
    # cnn.add(Dropout(0.5))
    # cnn.add(Dense(1))
    cnn.add(Dense(activation='sigmoid',units=1))

    # cnn.add(Dense(len(files), activation='softmax'))(cnn) 
    adam = Adam(learning_rate=0.0001)
    cnn.compile(optimizer=adam, loss='binary_crossentropy',metrics=['acc']) #// TODO check params doc model.compile()
    cnn.summary()
    
    return cnn

def LossOverEpochs(hist):
    plt.figure(figsize=(16, 9))
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Valid Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs fct')
    # plt.show()
# plotting train and validation curve LOSS
def train_validation_loss(cnn_model):
    train_loss = cnn_model.history['loss']
    val_loss = cnn_model.history['val_loss']
    fig = plt.figure(figsize = (8,5))
    plt.title("Loss over the epochs")
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel("Epochs", size=14)
    plt.ylabel('Loss')
    plt.grid(True, linestyle='-.')
    plt.legend()
    plt.show()
    return plt
    
# plotting train and validation curve ACC

def train_validation_acc(cnn_model):
    acc = cnn_model.history['acc']
    val_acc = cnn_model.history['val_acc']
    fig = plt.figure(figsize = (8,5))
    plt.title("Accuracy scores over the Epochs ")
    plt.plot(acc, label='training acc')
    plt.plot(val_acc, label='validation accuracy')
    plt.xlabel("Epochs", size=12)
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='-.')
    # plt.tick_params(labelcolor='r', labelsize='medium', width=3)
    plt.legend()
    # plt.show()
    return plt

def testaccuracy(cnn_model):
    plt.figure(figsize=(8, 6))
    plt.title('TEST Accuracy scores TEST')
    plt.plot(cnn_model.history['acc'])
    plt.plot(cnn_model.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    #plt.show()
    return plt
def train_accu(h):
    plt.style.use('ggplot')
    plt.figure()
  
    plt.plot(h.history["acc"], label='train_acc')
    plt.plot(h.history["val_acc"], label='val_acc')
    plt.title(' TRAINING accuracy ')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()      

from mlxtend.plotting import plot_confusion_matrix
def cm_plot_1(cnn):
    y_true = test_generator.classes
    Y_pred = cnn.predict(test_generator,steps=len(test_generator))
    y_pred = (Y_pred > 0.5).T[0]
    y_pred_prob = Y_pred.T[0]
    y_pred1 = np.argmax(Y_pred, axis=1)
    
    # predictions = cnn.predict(test_generator)
    
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # y_train_one_hot = one_hot_encoder.fit_transform(y_y_truetrue)
    # y_test_one_hot = one_hot_encoder.transform(y_true) 
    print('Reshaping X data')

    # predictions = one_hot_encoder.inverse_transform(Y_pred)
    cm2 = confusion_matrix(y_true,y_pred)
    print(classification_report(y_true, y_pred))
    
    # cm = confusion_matrix(y_true,y_pred,normalize = 'true') #// TODO:check
    # plot_confusion_matrix(cm2,figsize = (12,8), hide_ticks = True, cmap ='gray')

    # classnames = ['Bacteria', 'Normal', 'Pneumonia']
    # plt.figure(figsize=(8,8))
    # plt.title('Confusion matrix 2')
    sns.heatmap(cm2, cbar=False, xticklabels=y_true, yticklabels=y_true, fmt='d', annot=True, cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # plt.show()
    disp = ConfusionMatrixDisplay(cm2, display_labels=test_generator.classes)
    disp.from_predictions(y_true,y_pred)
    plt.title("Confusion Matrix", fontsize = 22)
     
    # plt.set_ticks(range(2),['Normal', 'Pneumonia'], fontsize = 16)
   
    plt.xticks(range(2), ['Normal','Pneumonia'], fontsize = 16)
    plt.yticks(range(2), ['Normal','Pneumonia'], fontsize = 16)
    # plt.show()
    

def ROC_curve_AUC_score(cnn):
    fig = plt.figure(figsize=(10, 8))
    y_true = test_generator.classes
    Y_pred = cnn.predict(test_generator, steps = len(test_generator))
    y_pred = (Y_pred > 0.5).T[0]
    y_pred_prob = Y_pred.T[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
    plt.plot(fpr, tpr, label='CNN (AUC = {:.2f}%)'.format(auc*100))
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='best')
    # plt.show()
    
def Summary_Stats(cnn):
    
    
    y_true = test_generator.classes
    Y_pred = cnn.predict(test_generator, steps= len(test_generator))
    y_pred = (Y_pred>0.5).T[0]
    y_pred_prob = Y_pred.T[0]
    cm = confusion_matrix(y_true, y_pred)
    
    TN, FP, FN, TP = cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
    #ravel, which is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array. 
    #The returned array has the same data type as the source array or input array.
    accuracy = (TP + TN) / np.sum(cm) 
    precision = TP / (TP+FP) 
    recall =  TP / (TP+FN)
    specificity = TN / (TN+FP) 
    f1 = 2*precision*recall / (precision + recall)
    stats_summary = '[Summary Statistics]\nAccuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Specificity = {:.2%} | F1 Score = {:.2%}'.format(accuracy, precision, recall, specificity, f1)
    print(f'{bcolors.OKCYAN}{stats_summary}{bcolors.ENDC}\n')


def allgraph(h):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(h.history["loss"], label='train_loss')
    plt.plot(h.history["val_loss"], label='val_loss')
    plt.plot(h.history["acc"], label='train_acc')
    plt.plot(h.history["val_acc"], label='val_acc')
    plt.title('MODEL TRAINING')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()  
    # plt.savefig("epochs.png")
    

def newtraingraph(modelfit):
    
    # plt.figure(figsize=(16, 9))
    plt.plot(modelfit.history['acc'])
    plt.plot(modelfit.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training set','Validation set'], loc='upper left')
    plt.show()

def training_graph( modelfit):
    
    plt.figure(figsize=(5, 6))
    plt.plot(modelfit.epoch, modelfit.history['acc'], label='accuracy')
    plt.title('Model Accuracy')
    plt.legend(['train'], loc='upper left')
    # plt.show()
    
    plt.figure(figsize=(5,6))
    plt.plot(modelfit.epoch, modelfit.history['loss'] , label='loss')
    plt.title('Model Loss')
    plt.legend(['train'], loc='upper left')
    # plt.show()

    plt.figure(figsize=(5, 6))
    plt.plot(modelfit.epoch, modelfit.history['val_acc'], label='val_accuracy')
    plt.title('Model Validation Accuracy')
    plt.legend(['train'], loc='upper left')
    # plt.show()

    plt.figure(figsize=(5, 6))
    plt.plot(modelfit.epoch, modelfit.history['val_loss'], label='val_loss')
    plt.title('Model Validation Loss')
    plt.legend(['train'], loc='upper left')
    # plt.show()
    
def Runner():
    
    load_data()
    
    cnn = model_1()
    # cnn.summary()
    
    # early_stop = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)   
    # rlr = ReduceLROnPlateau( monitor="val_accuracy",
                            # factor=0.01,
                            # patience=6,
                            # verbose=0,
                            # mode="max",
                            # min_delta=0.01)
                            
    # model_save = ModelCheckpoint('./stacked_model.h5',
    #                          save_best_only = True,
    #                          save_weights_only = False,
    #                          monitor = 'val_loss', 
    #                          mode = 'min', verbose = 1) 
    
    cnn_hist = cnn.fit(train_generator,epochs = 30, steps_per_epoch=len(train_generator), validation_data=val_generator, validation_steps=len(val_generator), verbose=1)# callbacks=[early_stop, model_save,rlr]
    
    # test_acc = cnn.evaluate(test_generator,steps=len(test_generator))
    
    # train_gen_len = len(train_generator)
    
    # print(f'{bcolors.OKGREEN} test accuracy is:\n',test_acc[1]*100,'%{bcolors.ENDC}')
    # print(f'{bcolors.HEADER}il y a {train_gen_len} iterations %{bcolors.ENDC}')
    
    # LossOverEpochs(cnn_hist) #// TODO:Train val loss + acc
    
    # cnn.evaluate(test_generator) #// TODO: A voir
    
    # cnn.predict()
    
    training_graph(cnn_hist) #//FIXME work
    
    # train_validation_loss(cnn_hist) #//FIXME work
    
    # train_validation_acc(cnn_hist) #//FIXME work
    # testaccuracy(cnn_hist) 
    train_accu(cnn_hist) 

    # cm_plot_1(cnn)   #//NOTE: run with train validation methode #//FIXME work
    
    # ROC_curve_AUC_score(cnn) #//TODO work : checker doc

    # Summary_Stats(cnn) #//FIXME work
    allgraph(cnn_hist)
    plt.show()
    
    
Runner()