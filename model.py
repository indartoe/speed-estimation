
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, ReLU
from tensorflow.keras.optimizers import Adam
from keras import backend as K


def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def CNNModel():
    model = Sequential()
    # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (240, 320, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Conv2D(36, kernel_size =(5, 5), strides=(2,2), kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Conv2D(48, kernel_size =(5, 5), strides=(2,2), kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size =(3, 3), strides = (1,1), kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Conv2D(64, kernel_size =(3, 3), strides= (1,1), padding = 'valid', kernel_initializer = 'he_normal'))
    model.add(Flatten())
    # model.add(ReLU())
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal'))
    # model.add(Dropout(0.5))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal'))
    # model.add(ReLU())
    model.add(ELU())
    model.add(Dense(1, kernel_initializer = 'he_normal'))

    adam = Adam(lr=1e-41)
    # model.compile(optimizer = adam, loss = 'mse')
    # model.compile(optimizer = adam, loss = ['mse', 'mae'], metrics=['mse', 'mae'])
    model.compile(optimizer = adam, loss = ['mae', rmse], metrics=[rmse, 'mae'])

    return model
