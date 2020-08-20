import os
import cv2
import consts
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

def resize(img,shape):
    return cv2.resize(img,shape)

def loadImage(img_path,shape):
    if '.DS_Store' not in img_path:
        return resize(cv2.imread(img_path),shape)
    else:
        return None

def OneHotEncode(x,classes):
    targets = []
    for i in x:
        target = [0] * classes
        target[i] = 1
        targets.append(target)
    return targets

def FromOneHot(OneHotEncodedArray):
    normal_array = []
    for target in OneHotEncodedArray:
        normal_array.append(list(target).index(max(list(target))))
    return np.array(normal_array)

def OneHotEncode_forone(x,classes):
    target = [0] * classes
    target[x] = 1
    return target

def make_read_for_input(path):
    img = cv2.resize(cv2.imread(path),consts.shape)
    return np.reshape(img,consts.shape_streamed_one)/255

def TrainTestSplit(X,Y,Shuffle):
    return train_test_split(X,Y,shuffle = Shuffle,test_size=0.1)

def getPrediction(l):
    return consts.CATS[list(l).index(max(list(l)))]
