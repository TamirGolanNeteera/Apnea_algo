import keras

from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv1D, GRU, LSTM, Bidirectional, MaxPool1D, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch

epochs = 500
batch_size = 128
samples_per_epoch = 320
sig = np.zeros([1500,1])

inp_sig = Input(shape=sig.shape)

x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(inp_sig)
x1 = BatchNormalization()(x1)
x1 = MaxPool1D(pool_size=4)(x1)

x2 = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(inp_sig)
x2 = BatchNormalization()(x2)
x2 = MaxPool1D(pool_size=4)(x2)

x3 = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(inp_sig)
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

x3 = Conv1D(filters=32,  kernel_size=7, activation='relu', padding='same')(x)
x3 = BatchNormalization()(x3)
x3 = MaxPool1D(pool_size=4)(x3)

x = Concatenate(axis=-1)([x1, x2, x3])
x = Dropout(0.3)(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool1D(pool_size=2)(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x) #BINARY classification S/W

model = keras.Model(inp_sig, output)
model.summary()