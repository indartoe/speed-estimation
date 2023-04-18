from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, ReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import keras

import tensorflow_model_optimization as tfmot
import numpy as np
import tensorflow as tf

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
 
def CNNModel():
    model = Sequential()
    # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    model.add(Conv2D(32, kernel_size =(3, 3), input_shape = (240, 320, 3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size =(3, 3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(1,1), padding='same'))
    model.add(Conv2D(128, kernel_size =(3, 3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(256, kernel_size =(3, 3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(1,1), padding='same'))
    model.add(Conv2D(512, kernel_size =(3, 3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(1024, kernel_size =(3, 3), strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(1,1), padding='same'))
    model.add(Flatten())
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(128))
    # model.add(Dropout(0.5))
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(64))
    model.add(ReLU())
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = ['mse', 'mae'], metrics=['mse', 'mae'])

    return model

    
