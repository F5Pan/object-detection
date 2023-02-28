# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:30:59 2022

@author: wayne
"""

import os
import cv2
from PIL import Image
import matplotlib.image as im
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv
import drugImages


x_train = []
y_train = []
z_train = []

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')#圖形標題
    plt.ylabel(train)#顯示y軸標籤
    plt.xlabel('Epoch')#設定x軸標籤是'Epoch'
    plt.legend(['train','validation'],loc='upper left')#設定圖例顯示'train','validation'在左上角
    plt.show()
# one_train_path = "D:/master/test2/"
# one_train_filedir = os.listdir(one_train_path)
#
# img = Image.open(one_train_path + 'a_0_225.jpg')
# img = np.array(img, np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow('img', img)
# data = np.array(img)
# cv2.imshow('data', data)
#
#
# s = img.shape
# print(s)
#
# n_rows = s[0]			# 图片行数
# n_cols = s[1]			# 图片列数
# n_channels = s[2]		# 图片的通道数
#
# print(n_rows)
# print(n_cols)
# print(n_channels)
#
# res = img.reshape([201,133,3])  # 将图片中原有的所有像素值 reshape 成 250 * 3000 的图
# cv2.imshow('res',res)
#
# cv2.waitKey(0)


# one_train_path = "D:/master/test1/"
# one_train_filedir = os.listdir(one_train_path)
#
# for file in one_train_filedir:
#     img = Image.open(one_train_path+file)
#     img = np.array(img, np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # data = np.array(img)
#     x_train.append(img)
#     y_train.append(0)
#
# one_train_path = "D:/master/test2/"
# one_train_filedir = os.listdir(one_train_path)
#
# for file in one_train_filedir:
#     img = Image.open(one_train_path+file)
#     img = np.array(img, np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # data = np.array(img)
#     x_train.append(img)
#     y_train.append(1)

# one_train_path = "D:/master/grayTriangleMintsRed/"
# one_train_filedir = os.listdir(one_train_path)
#
# for file in one_train_filedir:
#     img = Image.open(one_train_path+file)
#     data = np.array(img)
#     x_train.append(data)
#     y_train.append(2)
#
# one_train_path = "D:/master/grayCircleYellow/"
# one_train_filedir = os.listdir(one_train_path)
#
# for file in one_train_filedir:
#     img = Image.open(one_train_path+file)
#     data = np.array(img)
#     x_train.append(data)
#     y_train.append(3)


# x_train = np.array(x_train)


x_train,y_train,z_train = drugImages.read_drug_images(True)

print(x_train)
print(x_train.shape)

print(y_train)
print(z_train)
print(len(x_train))
x_train_normalize = x_train.reshape(-1,201,133,3).astype('float32')
x_train_normalize = x_train_normalize/255
y_train_onehot = np_utils.to_categorical(y_train)

print("Creating CNN model...")
# shape(高，寬，通道)
model= Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=(201,133,3)))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))


# Create CNN Model




print(model.summary())
# label沒有onehot用sparse_categorical_crossentropy
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# label有onehot用sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


train_history =model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.3, epochs=5, batch_size=20, verbose=2)
# 畫圖
# show_train_history(train_history,'accuracy','val_accuracy')
# show_train_history(train_history,'loss','val_loss')


model.save('drug.h5')
