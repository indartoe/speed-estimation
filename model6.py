import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, GlobalMaxPooling2D, DepthwiseConv2D, Softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers, activations
import tensorflow_addons as tfa
import math

# from keras_cv.layers import SqueezeAndExcite2D

def ChannelAttentionModule(input_tensor, reduction_ratio=16):
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2])
    shared_layer_one = Dense(units=input_tensor.shape[-1] // reduction_ratio, activation='relu', kernel_initializer='he_normal')(avg_pool)
    shared_layer_two = Dense(units=input_tensor.shape[-1] // reduction_ratio, activation='relu', kernel_initializer='he_normal')(max_pool)
    attention_avg = Dense(units=input_tensor.shape[-1], activation='sigmoid', kernel_initializer='glorot_uniform')(shared_layer_one)
    attention_max = Dense(units=input_tensor.shape[-1], activation='sigmoid', kernel_initializer='glorot_uniform')(shared_layer_two)
    channel_attention = layers.Multiply()([input_tensor, attention_avg + attention_max])
    return channel_attention

def SpatialAttentionModule(input_tensor, kernel_size=7):
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    combined = tf.concat([avg_pool, max_pool], axis=-1)
    attention = Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')(combined)
    spatial_attention = layers.Multiply()([input_tensor, attention])
    return spatial_attention

def CBAMModule(input_tensor):
    channel_attention = ChannelAttentionModule(input_tensor)
    spatial_attention = SpatialAttentionModule(input_tensor)
    cbam_features = layers.Add()([channel_attention, spatial_attention])
    return cbam_features

def ghost_conv(x, input_filter_size, stride):
    x1 = Conv2D(input_filter_size // 2, kernel_size=(3, 3), strides=stride, padding="same",
                       use_bias=False)(x)
    x2 = BatchNormalization(epsilon=1e-5)(x1)
    x2 = layers.Activation(activations.relu)(x2)
    x2 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same",
                                use_bias=False)(x2)
    return layers.Concatenate()([x1, x2])

def SE(x, input_filter_size, output_filter_size):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(input_filter_size//5)(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Dense(input_filter_size)(x)
    x = layers.Activation(activations.sigmoid)(x)
    return x

def fused_mb_conv(x, input_filter_size, output_filter_size, stride=1):
    shortcut = x
    if stride == 2:
        shortcut = layers.AveragePooling2D()(shortcut)
    if input_filter_size != output_filter_size:
        shortcut = ghost_conv(shortcut, output_filter_size, 1)
    # x = ghost_conv(x, output_filter_size, 1)
    # x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = Conv2D(input_filter_size, kernel_size =(3, 3), padding="same", strides=stride, use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    # x = CBAMModule(x)
    se = SE(x, input_filter_size, output_filter_size)
    x = layers.Multiply()([x, se])
    x = Conv2D(output_filter_size, kernel_size =(1, 1), padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    # x = CBAMModule(x)
    #start::these are part of feature fusion
    # x = ghost_conv(x, output_filter_size, 1)
    # x = layers.BatchNormalization(epsilon=1e-5)(x)
    if(stride==1):
        x = tfa.layers.StochasticDepth()([shortcut, x])
    #END::these are part of feature fusion
    return x
def mb_conv(x, input_filter_size, output_filter_size, stride=1):
    shortcut = x
    if stride == 2:
        shortcut = layers.AveragePooling2D()(shortcut)
    if input_filter_size != output_filter_size:
        shortcut = ghost_conv(shortcut, output_filter_size, 1)
    
    # x = ghost_conv(x, output_filter_size, 1)
    # x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = Conv2D(input_filter_size, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    x = DepthwiseConv2D(kernel_size =(3, 3), padding="same", strides=stride, use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    se = SE(x, input_filter_size, output_filter_size)
    x = layers.Multiply()([x, se])
    x = Conv2D(output_filter_size, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    # x = CBAMModule(x)
    #start::these are part of feature fusion
    # x = ghost_conv(x, output_filter_size, 1)
    # x = layers.BatchNormalization(epsilon=1e-5)(x)
    if(stride==1):
        x = tfa.layers.StochasticDepth()([shortcut, x])
    #END::these are part of feature fusion
    return x
def loggingParameter(loggingParameter: str):
    print(loggingParameter.replace("_", " ").capitalize()+ ":" + loggingParameter)
def CNNModel():
    # stride_1 = trial.suggest_categorical('stride_1', [1, 2])
    # stride_2 = trial.suggest_categorical('stride_2', [1, 2])
    # stride_3 = trial.suggest_categorical('stride_3', [1, 2])
    # stride_4 = trial.suggest_categorical('stride_4', [1, 2])
    # stride_5 = trial.suggest_categorical('stride_5', [1, 2])
    # stride_6 = trial.suggest_categorical('stride_6', [1, 2])
    # stride_7 = trial.suggest_categorical('stride_7', [1, 2])
    # parameterSuggestions = str(stride_1) + ";" + str(stride_2) + ";" + str(stride_3) + ";" + str(stride_4) + ";" + str(stride_5) + ";" + str(stride_6) + ";" + str(stride_7)
    # loggingParameter(str(stride_1))
    # loggingParameter(str(stride_2))
    # loggingParameter(str(stride_3))
    # loggingParameter(str(stride_4))
    # loggingParameter(str(stride_5))
    # loggingParameter(str(stride_6))
    # loggingParameter(str(stride_7))
    # model = Sequential()
    # # model.add(Conv2D(24, kernel_size =(5, 5), input_shape = (360, 640, 3), strides=(2,2), kernel_initializer = 'he_normal'))
    # model.add(Conv2D(24, kernel_size =(3, 3), input_shape = (60, 60, 1), strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    inp = tf.keras.Input(shape=(256, 256, 1))
    x = Conv2D(24, kernel_size =(3, 3), input_shape = (256, 256, 1), strides=2, padding="same", use_bias=False)(inp)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    # x = CBAMModule(x)
    # x = fused_mb_conv(x, 24, 48, 1)
    # x = CBAMModule(x)
    # x = fused_mb_conv(x, 48, 64, 2)
    # x = CBAMModule(x)
    # x = fused_mb_conv(x, 64, 128, 2)
    # x = CBAMModule(x)
    # x = fused_mb_conv(x, 128, 160, 2)
    # x = CBAMModule(x)
    # x = fused_mb_conv(x, 160, 176, 2)
    # x = CBAMModule(x)
    # x = mb_conv(x, 176, 304, 1)
    # x = mb_conv(x, 304, 512, 2)
    x = fused_mb_conv(x, 24, 48, 1)
    x = CBAMModule(x)
    x = fused_mb_conv(x, 48, 64, 2)
    x = CBAMModule(x)
    x = fused_mb_conv(x, 64, 128, 2)
    # x = CBAMModule(x)
    x = fused_mb_conv(x, 128, 160, 2)
    # x = CBAMModule(x)
    x = mb_conv(x, 160, 176, 2)
    # x = CBAMModule(x)
    x = mb_conv(x, 176, 304, 1)
    # x = CBAMModule(x)
    x = mb_conv(x, 304, 512, 2)
    # x = CBAMModule(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dense(1)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    # model.add(Conv2D(24, kernel_size =(3, 3), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(Conv2D(48, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))

    # model.add(Conv2D(48, kernel_size =(3, 3), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(Conv2D(64, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))

    # model.add(Conv2D(64, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # # model.add(GlobalAveragePooling2D())
    # model.add(Conv2D(128, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))

    # model.add(Conv2D(128, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # # model.add(GlobalAveragePooling2D())
    # model.add(Conv2D(160, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))

    # model.add(Conv2D(160, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # # model.add(GlobalAveragePooling2D())
    # model.add(Conv2D(256, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))

    # model.add(Conv2D(1280, kernel_size =(1, 1), padding="same", strides=(1,1), use_bias=False))
    # model.add(BatchNormalization(epsilon=1e-5))
    # model.add(layers.Activation(activations.swish))
    # model.add(GlobalAveragePooling2D())
    # # model.add(Flatten())
    # # model.add(ReLU())
    # model.add(Dense(512))
    # # model.add(Dropout(0.5))
    # model.add(ReLU())
    # model.add(Dense(256))
    # model.add(ReLU())
    # model.add(Dense(64))
    # model.add(ReLU())
    # model.add(Dense(1))

    rmsprop = RMSprop(lr=1e-4)
    # # model.compile(optimizer = adam, loss = 'mse')
    model.compile(optimizer = rmsprop, loss = ['mse', 'mae'], metrics=['mse', 'mae'])

    return model

CNNModel()