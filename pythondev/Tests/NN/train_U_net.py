import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import UpSampling1D, AveragePooling1D, Multiply, Add, GlobalAveragePooling1D, Activation, Conv2D, MaxPool2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import numpy as np

def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(16, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out = Multiply()([x_in, x])
    return x_out


def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x


def Unet(input_shape=(None, 1)):
    layer_n = 8
    kernel_size = 5
    depth = 2

    input_layer = Input(input_shape)
    input_layer_1 = AveragePooling1D(10)(input_layer)
    input_layer_2 = AveragePooling1D(30)(input_layer)

    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 2, 1)  # 1000
    #print("after 1st cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 5, 1)
    #print("after 2nd cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    #print("after 1st concat", x.shape)
    x = cbr(x, layer_n * 3, kernel_size, 3, 1)
    #print("after 3rd cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 3, kernel_size, 1)
    out_2 = x
    #print("out_2", x.shape)
    #print("before Concatenate", x.shape, input_layer_2.shape)
    x = Concatenate()([x, input_layer_2])
    #print("after 2nd concat", x.shape)
    x = cbr(x, layer_n * 4, kernel_size, 6, 1)
    #print("after 4th cbr", x.shape)
    #print(x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 4, kernel_size, 1)
    #print(x.shape)
    ########### Decoder
    x = UpSampling1D(6)(x)

    #print("before Concatenate", x.shape, out_2.shape)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n * 3, kernel_size, 1, 1)

    x = UpSampling1D(3)(x)
    #print("before Concatenate", x.shape, out_1.shape)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, kernel_size, 1, 1)

    # x = UpSampling1D(5)(x)
    # x = Concatenate()([x, out_0])
    # x = cbr(x, layer_n, kernel_size, 1, 1)

    # regressor
    # x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    # out = Activation("sigmoid")(x)
    # out = Lambda(lambda x: 12*x)(out)

    # classifier
    x = Conv1D(2, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = Activation("softmax")(x)

    model = Model(input_layer, out)

    return model

def MaskedUnet(input_shape=(None, 1)):
    layer_n = 32
    kernel_size = 7
    depth = 2

    input_layer = [Input(input_shape), Input(input_shape)]
    input_layer_1 = AveragePooling1D(10)(input_layer[0])
    input_layer_2 = AveragePooling1D(30)(input_layer[0])

    ########## Encoder
    x = cbr(input_layer[0], layer_n, kernel_size, 2, 1)  # 1000
    #print("after 1st cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 5, 1)
    #print("after 2nd cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    #print("after 1st concat", x.shape)
    x = cbr(x, layer_n * 3, kernel_size, 3, 1)
    #print("after 3rd cbr", x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 3, kernel_size, 1)
    out_2 = x
    #print("out_2", x.shape)
    #print("before Concatenate", x.shape, input_layer_2.shape)
    x = Concatenate()([x, input_layer_2])
    #print("after 2nd concat", x.shape)
    x = cbr(x, layer_n * 4, kernel_size, 6, 1)
    #print("after 4th cbr", x.shape)
    #print(x.shape)
    for i in range(depth):
        x = resblock(x, layer_n * 4, kernel_size, 1)
    #print(x.shape)
    ########### Decoder
    x = UpSampling1D(6)(x)

    #print("before Concatenate", x.shape, out_2.shape)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n * 3, kernel_size, 1, 1)

    x = UpSampling1D(3)(x)
    #print("before Concatenate", x.shape, out_1.shape)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, kernel_size, 1, 1)

    # x = UpSampling1D(5)(x)
    # x = Concatenate()([x, out_0])
    # x = cbr(x, layer_n, kernel_size, 1, 1)

    # regressor
    # x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    # out = Activation("sigmoid")(x)
    # out = Lambda(lambda x: 12*x)(out)

    # classifier
    x = Conv1D(2, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = Activation("softmax")(x)
    print(out.shape)
    print(input_layer[0].shape, input_layer[1].shape)
    out = Multiply()([out, input_layer[1]])
    model = Model(input_layer, out)

    return model
