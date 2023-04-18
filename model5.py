###############------------------------ BACKUP #######################################
# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D
# from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, ReLU
# from tensorflow.keras.optimizers import Adam
# from keras.metrics import RootMeanSquaredError
# import keras
 
# def CNNModel():
#     model = Sequential()
#     # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
#     model.add(Conv2D(64, kernel_size =(3, 3), input_shape = (240, 320, 3), strides=(2,2)))
#     model.add(ReLU())
#     model.add(Conv2D(64, kernel_size =(3, 3), strides=(2,2)))
#     model.add(ReLU())
#     model.add(Conv2D(64, kernel_size =(3, 3), strides=(2,2)))
#     model.add(Flatten())
#     model.add(ReLU())
#     model.add(Dense(64))
#     model.add(ReLU())
#     model.add(Dense(64))
#     # model.add(Dropout(0.5))
#     model.add(ReLU())
#     model.add(Dense(64))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))

#     adam = Adam(lr=1e-4)
#     model.compile(optimizer = adam, loss = ['mse', 'mae'], metrics=['mse', 'mae'])

#     return model



from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, ReLU
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
    model.add(Conv2D(64, kernel_size =(3, 3), input_shape = (240, 320, 3), strides=(2,2)))
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size =(3, 3), strides=(2,2)))
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size =(3, 3), strides=(2,2)))
    model.add(Flatten())
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(512))
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(512))
    # model.add(Dropout(0.5))
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(512))
    model.add(ReLU())
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = ['mse', 'mae'], metrics=['mse', 'mae'])

    return model

def Pruning(num_images, batch_size, epochs):
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(CNNModel(), **pruning_params)
    model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model_for_pruning

    
