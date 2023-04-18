
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Conv3D, ConvLSTM2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, MaxPooling3D, Input, TimeDistributed, Convolution2D
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from tensorflow import keras
# from ugs_utils.keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)



def CNNModel():
        # input_shape = (256, 240, 320, 3)
        inp = Input(shape=(None, 240, 320, 3))

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
        )(inp)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
        )(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(
                filters=64,
                kernel_size=(1, 1),
                padding="same",
                return_sequences=True,
                activation="relu",
                )(x)
        x = Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
                )(x)
        
        model = keras.models.Model(inp, x)
        model.compile(
                loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
        )
        return model
#     input_img = Input((256, 240, 320, 3), name='input')
#     x = ConvLSTM2D(nb_filter=12, nb_row = 3, nb_col = 3, border_mode = 'same', return_sequences=True)(input_img)
#     output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)
#     model = Sequential()
#     model.add(ConvLSTM2D(filters=12, kernel_size=(1, 2),
#                 dropout=0.1, activation='relu', return_sequences=True, input_shape=(None, 240, 320, 3)))
#     model.add(ConvLSTM2D(filters=12, kernel_size=(
#             1, 2), dropout=0.1, activation='relu', return_sequences=True))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(1, name='out_layer', activation="linear"))
    # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    # model.add(ConvLSTM2D(
    #     filters=64,
    #     kernel_size=(5, 5),
    #     input_shape=(128, 240, 320, 3),
    #     padding='same',
    #     activation="relu",
    #     return_sequences=True))
    # # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(ConvLSTM2D(
    #     filters=96,
    #     kernel_size=(3, 3),
    #     padding='same',
    #     activation="relu",
    #     return_sequences=True))
    # model.add(ConvLSTM2D(24, kernel_size =(5, 5), input_shape = (128, 240, 320, 3), strides=(2,2), kernel_initializer = 'he_normal', return_sequences=True, data_format='channels_last'))
    # model.add(ELU())
    # model.add(ConvLSTM2D(36, kernel_size =(5, 5), strides=(2,2), kernel_initializer = 'he_normal', return_sequences=True, data_format='channels_last'))
    # model.add(ELU())
    # model.add(ConvLSTM2D(48, kernel_size =(5, 5), strides=(2,2), kernel_initializer = 'he_normal', return_sequences=True, data_format='channels_last'))
    # model.add(ELU())
    # model.add(Dropout(0.5))
    # model.add(ConvLSTM2D(64, kernel_size =(3, 3), strides = (1,1), kernel_initializer = 'he_normal', return_sequences=True, data_format='channels_last'))
    # model.add(ELU())
    # model.add(ConvLSTM2D(64, kernel_size =(3, 3), strides= (1,1), padding = 'valid', kernel_initializer = 'he_normal', return_sequences=True, data_format='channels_last'))
    # model.add(Flatten())
    # model.add(ELU())
    # model.add(Dense(100, kernel_initializer = 'he_normal'))
    # model.add(ELU())
    # model.add(Dense(50, kernel_initializer = 'he_normal'))
    # # model.add(Dropout(0.5))
    # model.add(ELU())
    # model.add(Dense(10, kernel_initializer = 'he_normal'))
    # model.add(ELU())
    # model.add(Dense(1, kernel_initializer = 'he_normal'))
    # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # model.add(Flatten())

    # model.add(Dense(32, kernel_initializer='TruncatedNormal'))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.2))

    # model.add(Dense(16, kernel_initializer='TruncatedNormal'))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.2))

    # model.add(Dense(1))
#     model.summary()
    # adam = Adam(lr=1e-4)
#     model.compile(loss='binary_crossentropy', optimizer='adam')

#     return model
