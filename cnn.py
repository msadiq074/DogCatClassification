# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:27:35 2020

@author: msadi
"""

# Part I - Building the CNN

# import the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters=32, kernel_size=[3,3],input_shape=(64,64,3), activation = 'relu'))

# Step 2 - MaxPooling
classifier.add(MaxPooling2D( pool_size=(2,2) ) )

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image Data Generator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset3/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset3/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit(training_set,
          steps_per_epoch=1998,
          epochs=3,
          validation_data=test_set,
          validation_steps=400)


# Make new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path='dataset3/single_prediction/cat_or_dog_1.jpg',target_size=[64,64])
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
pred = classifier.predict(test_image)
training_set.class_indices
test_image = image.load_img(path='dataset3/single_prediction/cat_or_dog_2.jpg',target_size=[64,64])
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
pred2 =classifier.predict(test_image)
pred2
