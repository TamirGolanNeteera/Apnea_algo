#from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/')
sys.path.append(conf_path + '/Tests/NN')
import os
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import keras
# from tensorflow import keras
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import tensorflow as tf
import keras
from keras import layers
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
#from keras.engine.topology import Layer
import matplotlib.colors as mcolors
import numpy as np
from Tests.vsms_db_api import DB
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Add
#from keras.engine.topology import Layer


def smaller_vgg_model_1d(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(inp_sig)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)


def small_vgg_model_1d(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(inp_sig)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=2)(x1)
    x2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=2)(x2)
    x3 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=2)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(64, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)

def small_vgg_model(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)

def large_vgg_model_late_batchnorm(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)3
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)


def large_resnet_model_late_batchnorm(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)




def large_resnet_model_late_batchnorm_1D(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    print('reach here AA')
    print(inp_sig)
    print(__name__)
    print(__file__)
    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inp_sig)
    print('reach here AAA')
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    print('reach here AAAA')
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    print('reach here B')
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    print('reach here C')
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)
    print('reach here?')

    return keras.Model(inp_sig, output)





def vs_resnet_model_late_batchnorm_1D(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(64, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)





def large_resnet_model_late_batchnorm_1D_binary_classifier(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool1D(pool_size=2)(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv1D(filters=32, kernel_size=3, padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='sigmoid')(x2)

    return keras.Model(inp_sig, output)



def large_resnet_encoder(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    return x



def increment_resnet_model_late_batchnorm(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv2D(filters=16, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)


def large_resnet_model_late_batchnorm_multiple_outputs(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    output = Dense(8, activation='relu')(x2)

    return keras.Model(inp_sig, output)



def small_resnet_model_late_batchnorm(input_shape):

    print(input_shape)
    inp_sig = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    x = MaxPool2D(pool_size=(2, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(1, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(1, 2))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = MaxPool2D(pool_size=(2, 1))(x)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    first_layer = Activation("linear", trainable=False)(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(first_layer)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding="same")(x)
    residual = Add()([x, first_layer])
    x = Activation("relu")(residual)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x2 = Flatten()(x)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)

def large_vgg_model_late_batchnorm_2_outputs(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)3
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    x1 = Dense(32, activation='relu')(x2)
    output1 = Dense(1, activation='relu')(x1)
    output2 = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, [output1, output2])

def large_vgg_model_late_batchnorm_for_bioid(input_shape, n):
    inp_sig = Input(shape=input_shape)
    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(inp_sig)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(inp_sig)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(n, activation='softmax')(x2)

    return keras.Model(inp_sig, output)

def large_vgg_model_late_batchnorm_classification(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)3
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(10, activation='softmax')(x2)

    return keras.Model(inp_sig, output)


def large_vgg_model_late_batchnorm_1d(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)3
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool1D(pool_size=4)(x1)
    x2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool1D(pool_size=4)(x2)
    x3 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool1D(pool_size=4)(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)
    return keras.Model(inp_sig, output)


def large_vgg_model_late_batchnorm_test_time_dropout(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same', strides=(2,1))(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(5,1), activation='relu', padding='same', strides=(2,1))(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same', strides=(2,1))(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x, training=True)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same', strides=(2,1))(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same', strides=(2,1))(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same', strides=(2,1))(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x, training=True)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same', strides=(2,1))(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same', strides=(2,1))(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same', strides=(2,1))(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x, training=True)

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same', strides=(2,1))(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same', strides=(2,1))(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same', strides=(2,1))(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x, training=True)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)



def large_vgg_model_late_batchnorm_more_pooling(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)

def very_large_vgg_model_late_batchnorm(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)
def large_vgg_model_no_batchnorm(input_shape):
    inp_sig = Input(shape=input_shape)
    x1 = Conv2D(filters=8, kernel_size=(3,1), activation='relu', padding='same')(inp_sig)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4,2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4,2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7,1), activation='relu', padding='same')(inp_sig)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3,1), activation='relu', padding='same')(x)
    #x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2,2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5,1), activation='relu', padding='same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2,2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7,1), activation='relu', padding='same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2,2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x2 = Flatten()(x)

    x2 = Dense(32, activation='relu')(x2)
    output = Dense(1, activation='relu')(x2)

    return keras.Model(inp_sig, output)

#####################################################################################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    print(head_size, num_heads)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

#The main part of our model is now complete. We can stack multiple of those transformer_encoder blocks and we can also proceed to add the final Multi-Layer Perceptron classification head. Apart from a stack of Dense layers, we need to reduce the output tensor of the TransformerEncoder part of our model down to a vector of features for each data point in the current batch. A common way to achieve this is to use a pooling layer. For this example, a GlobalAveragePooling1D layer is sufficient.

def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    x = Conv2D(filters=16, kernel_size=(1,8), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(1,8))(x)
    x = Conv2D(filters=16, kernel_size=(11,1), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2,1))(x)
    x = Conv2D(filters=16, kernel_size=(7,1), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2,1))(x)
    x = Conv2D(filters=16, kernel_size=(3, 1), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 1))(x)


    x = Reshape(target_shape=(625, 16))(x)
    temp_model = keras.Model(inputs, x)
    temp_model.summary()
    print(x.shape)
    for _ in range(num_transformer_blocks):
        print("shape = ", x.shape)
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        print("encoded shape", x.shape)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    return keras.Model(inputs, outputs)

########################################################################


embed_size = 60
import keras.backend as K

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


def CnnTransformerModel():
    i = Input(shape=(5000, 8))

    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(i)

    x = Convolution1D(8, kernel_size=10, strides=1, activation='relu')(i)
    x = MaxPool1D(pool_size=2)(x)

    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(16, kernel_size=10, strides=1, activation='relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(32, kernel_size=10, strides=1, activation='relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(x)

    x = Convolution1D(64, kernel_size=10, strides=1, activation='relu')(x)
    x = MaxPool1D(pool_size=2)(x)
    print(x.shape)
    x = Bidirectional(LSTM(128, return_sequences=True, return_state=False))(x)

    x = Bidirectional(LSTM(64, return_sequences=True, return_state=False))(x)

    x, slf_attn = MultiHeadAttention(n_head=5, d_model=300, d_k=64, d_v=64, dropout=0.3)(x, x, x)

    avg_pool = GlobalAveragePooling1D()(x)

    avg_pool = Dense(60, activation='relu')(avg_pool)

    y = Dense(1, activation='relu')(avg_pool)
    print(y.shape)
    return Model(inputs=[i], outputs=[y])


def MyCnnTransformerModel():
    inp_sig = Input(shape=(5000, 8, 1))

    x1 = Conv2D(filters=8, kernel_size=(3, 1), activation='relu', padding='same')(inp_sig)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(4, 2))(x1)
    x2 = Conv2D(filters=8, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(4, 2))(x2)
    x3 = Conv2D(filters=8, kernel_size=(7, 1), activation='relu', padding='same')(inp_sig)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(4, 2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=16, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x2 = Conv2D(filters=16, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 2))(x2)
    x3 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dropout(0.3)(x)

    x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 2))(x1)
    x2 = Conv2D(filters=32, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 2))(x2)
    x3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 2))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])

    x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size=(2, 1))(x1)
    x2 = Conv2D(filters=64, kernel_size=(5, 1), activation='relu', padding='same')(x)
    # x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size=(2, 1))(x2)
    x3 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(x)
    # x3 = BatchNormalization()(x3)
    x3 = MaxPool2D(pool_size=(2, 1))(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Reshape((156,192))(x)
    print(x.shape)
    x = Bidirectional(LSTM(128, return_sequences=True, return_state=False))(x)

    x = Bidirectional(LSTM(64, return_sequences=True, return_state=False))(x)

    #x, slf_attn = MultiHeadAttention(n_head=5, d_model=300, d_k=64, d_v=64, dropout=0.3)(x, x, x)

    #avg_pool = GlobalAveragePooling1D()(x)

    x = Dense(60, activation='relu')(x)

    y = Dense(1, activation='relu')(x)
    print(y.shape)
    return Model(inputs=[inp_sig], outputs=[y])

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res
# The main part of our model is now complete. We can stack multiple of those transformer_encoder blocks and we can also proceed to add the final Multi-Layer Perceptron classification head. Apart from a stack of Dense layers, we need to reduce the output tensor of the TransformerEncoder part of our model down to a vector of features for each data point in the current batch. A common way to achieve this is to use a pooling layer. For this example, a GlobalAveragePooling1D layer is sufficient.

def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation='relu')(x)

    return keras.Model(inputs, outputs)
# Train and evaluate

############################=========================================================
import tensorflow as tf

import keras
from keras import layers
#import tensorflow_addons as tfa

import numpy as np

import math



NUM_PATCHES_VS = 50
NUM_PATCHES_BP = 100
#print("NUM_PATCHES", NUM_PATCHES)
# ViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]

# TOKENLEARNER
NUM_TOKENS = 8


def position_embedding(
    projected_patches, num_patches, projection_dim=PROJECTION_DIM
):
    # Build the positions.
    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    print(projected_patches.shape)
    print(encoded_positions.shape)
    return projected_patches + encoded_positions

# MLP block for Transformer
# This serves as the Fully Connected Feed Forward block for our Transformer.

def mlp(x, dropout_rate, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# TokenLearner module
# The following figure presents a pictorial overview of the module (source).
# TokenLearner module GIF
# The TokenLearner module takes as input an image-shaped tensor. It then passes it through multiple single-channel convolutional layers extracting different spatial attention maps focusing on different parts of the input. These attention maps are then element-wise multiplied to the input and result is aggregated with pooling. This pooled output can be trated as a summary of the input and has much lesser number of patches (8, for example) than the original one (196, for example).

# Using multiple convolution layers helps with expressivity. Imposing a form of spatial attention helps retain relevant information from the inputs. Both of these components are crucial to make TokenLearner work, especially when we are significantly reducing the number of patches.

def token_learner(inputs, number_of_tokens=NUM_TOKENS):
    # Layer normalize the inputs.
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(inputs)  # (B, H, W, C)

    # Applying Conv2D => Reshape => Permute
    # The reshape and permute is done to help with the next steps of
    # multiplication and Global Average Pooling.
    attention_maps = keras.Sequential(
        [
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            # This conv layer will generate the attention maps
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",  # Note sigmoid for [0, 1] output
                padding="same",
                use_bias=False,
            ),
            # Reshape and Permute
            layers.Reshape((-1, number_of_tokens)),  # (B, H*W, num_of_tokens)
            layers.Permute((2, 1)),
        ]
    )(
        x
    )  # (B, num_of_tokens, H*W)

    # Reshape the input to align it with the output of the conv block.
    num_filters = inputs.shape[-1]
    inputs = layers.Reshape((1, -1, num_filters))(inputs)  # inputs == (B, 1, H*W, C)

    # Element-Wise multiplication of the attention maps and the inputs
    attended_inputs = (
        attention_maps[..., tf.newaxis] * inputs
    )  # (B, num_tokens, H*W, C)

    # Global average pooling the element wise multiplication result.
    outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
    return outputs

#Transformer block
def transformer(encoded_patches):
    # Layer normalization 1.
    print("encoded_patches.shape", encoded_patches.shape)
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("x1.shape", x1.shape)
    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    print("attention_output.shape", attention_output.shape)
    print("encoded_patches.shape", encoded_patches.shape)
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1)

    # Skip connection 2.
    print("layers.Add()([attention_output, encoded_patches]).shape", x2.shape)
    print("mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1).shape", x4.shape)
    encoded_patches = layers.Add()([x4, x2])
    print("encoded_patches = = layers.Add()([x4, x2]).shape", encoded_patches.shape)
    return encoded_patches

#ViT model with the TokenLearner module
def create_vit_BP_regressor_1d(input_shape, use_token_learner=True, token_learner_units=NUM_TOKENS):
    inputs = layers.Input(shape=input_shape)  # (B, H, W, C)

    print(inputs.shape)

    projected_patches = layers.Conv2D(
        filters=PROJECTION_DIM,
        kernel_size=(25,1),#kernel_size=50,
        strides=(25,8),#strides=50,
        padding="SAME",
    )(inputs)
    print(projected_patches.shape)
    _, w, h, c = projected_patches.shape
    projected_patches = layers.Reshape((w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(projected_patches, 2*NUM_PATCHES_BP)  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS):
        # Add a Transformer block.
        print("transformer block", i)
        encoded_patches = transformer(encoded_patches)

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == NUM_LAYERS // 2:
            _, hh, c = encoded_patches.shape
            print( hh, c)

            print("reshaping to patches")

            encoded_patches = layers.Reshape((20, 10, c))(encoded_patches)
            print("encoded_patches.shape", encoded_patches.shape)
            #encoded_patches = layers.Reshape((h, h, c))(
            #    encoded_patches
            #)  # (B, h, h, projection_dim)
            encoded_patches = token_learner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("representation.shape after layerNorm", representation.shape)
    representation = layers.GlobalAvgPool1D()(representation)
    print("representation.shape, after GAP", representation.shape)
    # Classify outputs.
    outputs = layers.Dense(1, activation="relu")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_vit_VS_regressor_1d(input_shape, use_token_learner=True, token_learner_units=NUM_TOKENS):
    inputs = layers.Input(shape=input_shape)  # (B, H, W, C)

    print(inputs.shape)

    projected_patches = layers.Conv1D(
        filters=PROJECTION_DIM,
        kernel_size=50,#kernel_size=50,
        strides=50,#strides=50,
        padding="SAME",
    )(inputs)
    print(projected_patches.shape)

    print(projected_patches.shape)
    h, w, c = projected_patches.shape
#    projected_patches = layers.Reshape((h * w, c))(
    projected_patches = layers.Reshape((w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(projected_patches, NUM_PATCHES_VS)  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS):
        # Add a Transformer block.
        print("transformer block", i)
        encoded_patches = transformer(encoded_patches)

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == NUM_LAYERS // 2:
            _, hh, c = encoded_patches.shape
            print( hh, c)

            print("reshaping to patches")

            encoded_patches = layers.Reshape((10, 5, c))(encoded_patches)
            print("encoded_patches.shape", encoded_patches.shape)
            #encoded_patches = layers.Reshape((h, h, c))(
            #    encoded_patches
            #)  # (B, h, h, projection_dim)
            encoded_patches = token_learner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("representation.shape after layerNorm", representation.shape)
    representation = layers.GlobalAvgPool1D()(representation)
    print("representation.shape, after GAP", representation.shape)
    # Classify outputs.
    outputs = layers.Dense(1, activation="relu")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_vit_BP_regressor_8d(input_shape, use_token_learner=True, token_learner_units=NUM_TOKENS):
    inputs = layers.Input(shape=input_shape)  # (B, H, W, C)

    print(inputs.shape)

    projected_patches = layers.Conv2D(
        filters=PROJECTION_DIM,
        kernel_size=(50,8),#kernel_size=50,
        strides=(50,8),#strides=50,
        padding="SAME",
    )(inputs)
    print(projected_patches.shape)
    _, w, h, c = projected_patches.shape
    projected_patches = layers.Reshape((w*h, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(projected_patches, NUM_PATCHES_BP, projection_dim=PROJECTION_DIM)  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS):
        # Add a Transformer block.
        print("transformer block", i)
        encoded_patches = transformer(encoded_patches)

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == NUM_LAYERS // 2:
            _, hh, c = encoded_patches.shape
            print( hh, c)

            print("reshaping to patches")

            encoded_patches = layers.Reshape((10, 10, c))(encoded_patches)
            print("encoded_patches.shape", encoded_patches.shape)
            #encoded_patches = layers.Reshape((h, h, c))(
            #    encoded_patches
            #)  # (B, h, h, projection_dim)
            encoded_patches = token_learner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("representation.shape after layerNorm", representation.shape)
    representation = layers.GlobalAvgPool1D()(representation)
    print("representation.shape, after GAP", representation.shape)
    # Classify outputs.
    outputs = layers.Dense(1, activation="relu")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
