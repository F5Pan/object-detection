# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:30:59 2022

@author: wayne
"""

import os
from PIL import Image
import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils import np_utils


x_train = []
y_train = []

one_train_path = "D:/master/AA/"
one_train_filedir = os.listdir(one_train_path)

for file in one_train_filedir:
    img = Image.open(one_train_path+file)
    data = np.array(img)
    x_train.append(data)
    y_train.append(0)

one_train_path = "D:/master/grayTriangleMintsRed/"
one_train_filedir = os.listdir(one_train_path)

for file in one_train_filedir:
    img = Image.open(one_train_path+file)
    data = np.array(img)
    x_train.append(data)
    y_train.append(1)

one_train_path = "D:/master/grayCircleYellow/"
one_train_filedir = os.listdir(one_train_path)

for file in one_train_filedir:
    img = Image.open(one_train_path+file)
    data = np.array(img)
    x_train.append(data)
    y_train.append(2)

one_train_path = "D:/master/grayCapsuleBlue/"
one_train_filedir = os.listdir(one_train_path)

for file in one_train_filedir:
    img = Image.open(one_train_path+file)
    data = np.array(img)
    x_train.append(data)
    y_train.append(3)

    
x_train = np.array(x_train)
print(type(x_train))
print(x_train.shape)
x_train_normalize = x_train.reshape(120,300,300,3).astype('float32')
x_train_normalize = x_train_normalize/255
y_train_onehot = np_utils.to_categorical(y_train)

input_shape = (300, 300, 3)
model = Sequential([
    Conv2D(16, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same', ),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same', ),
    Conv2D(64, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.5, epochs=150, batch_size=10, verbose=2)
model.save('drug.h5')
