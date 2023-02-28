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



def predict(image):
    # image_path ="D:/master/images/0/c_0_492.jpg"
    # image = cv.imread(image_path)
    reImage = cv.resize(image, (133, 201))
    cv.imshow('reImage',reImage)
    normalizer_image = reImage / 255.0
    cv.imshow('normalizer_image',normalizer_image)


    # imageCvt = cv.cvtColor(reImage,cv.COLOR_BGR2GRAY)
    # grayImage = imageCvt
    # cv.imshow('number',grayImage)
    testImage = normalizer_image.reshape(1,201,133,3).astype('float32')



    model = load_model('drug.h5')
    resultRate = model.predict(testImage)
    result = np.argmax(resultRate,axis =1)
    resultRate1 = '普福分 = {:.2f},咳嗽藥= {:.2f}'.format(resultRate[0][0],resultRate[0][1])

    # print(resultRate)
    # print(resultRate[0][0])
    # print(resultRate[0][1])
    # print(resultRate[0][2])
    print(resultRate1)
    # print(result)

    rate = resultRate[0][0]
    rate1 = resultRate[0][1]
    # rate2 = resultRate[0][2]
    # rate3 =resultRate[0][3]
    print(rate)
    print(rate1)
    # print(rate2)
    # print(rate3)
    if (rate>0.8):
        print('普福芬')
        str = 'orange'
    elif (rate1>0.8):
        print('咳嗽藥')
        str = 'red'
    # elif (rate2>0.8):
    #     print('是圓形')
    # elif (rate2>0.8):
    #     print('是膠囊')
    # else :
    #     print('都不是')

    # cv.waitKey(0)
    # x = model.predict_classes(testImage)
    #
    # print(x)

    return str

if __name__ == '__main__':
    predict()