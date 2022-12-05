import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import cv2


image_dims = 128
batch_size = 64

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(image_dims,image_dims,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())  

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])

input_path = './chest_Xray/'

training_data_gen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
train_gen = training_data_gen.flow_from_directory(directory=input_path+'train/', target_size=(image_dims, image_dims), batch_size=batch_size, class_mode='binary')

validation_data_generator = ImageDataGenerator(rescale= 1./255)
validation_gen = validation_data_generator.flow_from_directory(directory= input_path+ 'val/',
                             target_size=(image_dims,image_dims),
                             batch_size= batch_size,
                             class_mode= 'binary')

epochs = 2

history = model.fit(train_gen,
                                steps_per_epoch=4,
                                epochs=epochs,
                                validation_data=validation_gen,
                                validation_steps= validation_gen.samples
                              )


plt.figure(figsize=(8,6))
plt.title('TechVidvan Accuracy scores')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()