# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:21:59 2022

@author: wayne
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.utils import np_utils
import cv2 as cv
from keras.models import load_model


# image_path ="D:/master/grayCapsuleBlue/b5.jpg"
image_path ="D:/master/AA/a1.jpg"
image = cv.imread(image_path)
reImage = cv.resize(image,(300,300))

# imageCvt = cv.cvtColor(reImage,cv.COLOR_BGR2GRAY)
# grayImage = imageCvt
# cv.imshow('number',grayImage)
testImage = reImage.reshape(1,300,300,3).astype('float32')/255


model = load_model('drug.h5')
resultRate = model.predict(testImage)
result = np.argmax(resultRate,axis =1)
resultRate1 = '三角 = {:.2f},圓形= {:.2f},膠囊= {:.2f}'.format(resultRate[0][0],resultRate[0][1],resultRate[0][2],resultRate[0][3])
# resultRate1 = '三角 = {:.2f},圓形= {:.2f}'.format(resultRate[0][0],resultRate[0][1])

# print(resultRate)
# print(resultRate[0][0])
# print(resultRate[0][1])
# print(resultRate[0][2])
print(resultRate1)
# print(result)

rate = resultRate[0][0]
rate1 = resultRate[0][1]
rate2 = resultRate[0][2]
rate3 =resultRate[0][3]
print(rate)
print(rate1)
print(rate2)
print(rate3)
if (rate>0.8):
    print('粉紅膠囊')
elif (rate1>0.8):
    print('是三角形')
elif (rate2>0.8):
    print('是圓形')
elif (rate2>0.8):
    print('是膠囊')
else :
    print('都不是')

cv.waitKey(0)
# x = model.predict_classes(testImage)
#
# print(x)