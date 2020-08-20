from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras.optimizers import SGD,Adam
from keras.models import Sequential

import consts

def getModel():
    
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU())

        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU())


        model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(axis=3))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(axis=3))
        model.add(LeakyReLU())

        model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(axis=3))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(axis=3))
        model.add(LeakyReLU())


        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(consts.classes, activation='softmax'))


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

        return model
