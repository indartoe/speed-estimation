from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, ReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import keras

import tensorflow_model_optimization as tfmot
import numpy as np
import tensorflow as tf

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
import visualkeras
from PIL import ImageFont
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

def loggingParameter(loggingParameter: str):
    print(loggingParameter.replace("_", " ").capitalize()+ ":" + loggingParameter)
 
parameterSuggestions = ""
def CNNModel():
    
    # num_filters_1 = trial.suggest_categorical('num_filters_1', [32])
    # num_filters_2 = trial.suggest_categorical('num_filters_2', [32])
    # num_filters_3 = trial.suggest_categorical('num_filters_3', [32, 64])
    # num_filters_4 = trial.suggest_categorical('num_filters_4', [32, 64])
    # num_filters_5 = trial.suggest_categorical('num_filters_5', [32, 64])
    # num_filters_6 = trial.suggest_categorical('num_filters_6', [32, 64, 128])
    # max_pool_1 = trial.suggest_categorical('max_pool_1', [2, 3])
    # max_pool_2 = trial.suggest_categorical('max_pool_2', [2,3])
    
    # parameterSuggestions = str(num_filters_1) + ";" + str(num_filters_2) + ";" + str(num_filters_3) + ";" + str(num_filters_4) + ";" + str(num_filters_5) + ";" + str(num_filters_6) 
    #     # + ";" + max_pool_1 + ";" + max_pool_2
    # print("####Parameters:######")
    # loggingParameter(str(num_filters_1))
    # loggingParameter(str(num_filters_2))
    # loggingParameter(str(num_filters_3))
    # loggingParameter(str(num_filters_4))
    # loggingParameter(str(num_filters_5))
    # loggingParameter(str(num_filters_6))
    # loggingParameter(str(max_pool_1))
    # loggingParameter(str(max_pool_2))
    # print("####END::PARAMETER######")
    model = Sequential()
    # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    model.add(Conv2D(32, kernel_size =(3, 3), input_shape = (60, 60, 1 ), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size =(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding= "same"))
    model.add(Conv2D(64, kernel_size =(3, 3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size =(3, 3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"))
    model.add(Conv2D(128, kernel_size =(3, 3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(128, kernel_size =(3, 3), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"))
    model.add(Flatten())
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    # model.add(Dense(4096))
    model.add(Dense(64))
    # model.add(Dropout(0.5))
    model.add(ReLU())
    # model.add(Dense(4096))
    # model.add(Dense(64))
    # model.add(ReLU())
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = ['mse', 'mae'], metrics=['mse', 'mae'])

    # return model, parameterSuggestions
    return model

# font = ImageFont.truetype("arial.ttf", 32)
# model = CNNModel()
# model.summary()
# visualkeras.layered_view(model, legend=True, to_file='data/model_output.png', font=font) # write to disk