import os
import cv2
import data
import tools
import consts
import model as md
import numpy as np


from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD,Adam
from keras.models import Sequential

x_train,x_test,y_train,y_test = data.getData()

model = md.getModel()

## training the model
model.fit(x_train,y_train,epochs=consts.epochs,shuffle=True,validation_data=(x_test,y_test))

prediction = tools.FromOneHot(model.predict(x_test))
correct = tools.FromOneHot(y_test)


acc = str(round((np.sum(prediction == correct)/len(list(y_test))) * 100))
print('Acc :' + acc + '%')

model.save('ASL' + acc + '.h5')
