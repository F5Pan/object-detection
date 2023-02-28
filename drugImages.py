from sklearn.utils import shuffle
import numpy as np
import os
import cv2

def read_image(path):
    image = cv2.imread(path)
    # resize(圖,(寬,高))
    image = cv2.resize(image, (133, 201))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image',image)
    # normalizer_image = image / 255.0
    # cv2.imshow('normalizer_image',normalizer_image)

    return image


def read_drug_images(training=True):
    drug_dir = "D:/master/images/drug"

    if training:
        red = drug_dir + "/0/"
        yellow = drug_dir + "/1/"
    else:
        red = drug_dir + "test/0/"
        yellow = drug_dir + "test/1/"

    images = []
    labels = []
    image_name = []

    for f in os.listdir(red):
        images.append(read_image(red + f))
        labels.append(0)
        image_name.append(f)

    for f in os.listdir(yellow):
        images.append(read_image(yellow + f))
        labels.append(1)
        image_name.append(f)

    return shuffle(np.array(images), np.array(labels), np.array(image_name))


if __name__ == '__main__':

    x_train, y_train, z_train = read_drug_images(True)
    cv2.imshow('train',x_train[1])
    cv2.waitKey(0)
