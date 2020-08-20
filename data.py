import os
import cv2
import tools
import consts
import numpy as np

def getData():
    X = []
    Y = []

    print('Started loading Data')

    for path in consts.PATHS:

        label = consts.PATHS.index(path)
        target = tools.OneHotEncode_forone(label,consts.classes)

        for image_name in os.listdir(path):

            file = os.path.join(path,image_name)
            image = tools.loadImage(file,consts.shape)
            if image is not None:
                X.append(image)
                Y.append(target)

    print('Data Prepared')

    return tools.TrainTestSplit(np.array(X)/255.0,np.array(Y),True)

def saveData(X,Y):
    np.save('x',X)
    np.save('y',Y)

# def loadData():
    # return tools.TrainTestSplit(np.load('x.npy'),np.load('y.npy'),True)
