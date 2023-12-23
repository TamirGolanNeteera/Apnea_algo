import keras

from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate
from keras.layers import Conv1D, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout

def ahi_model(inp_sig):

    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', name='conv1a')(inp_sig)
    x1 = BatchNormalization(name='bn1a')(x1)
    x1 = MaxPool1D(pool_size=4, name='maxpool1a')(x1)

    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same', name='conv1b')(inp_sig)
    x2 = BatchNormalization(name='bn1b')(x2)
    x2 = MaxPool1D(pool_size=4, name='maxpool1b')(x2)

    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same', name='conv1c')(inp_sig)
    x3 = BatchNormalization(name='bn1c')(x3)
    x3 = MaxPool1D(pool_size=4, name='maxpool1c')(x3)

    x = Concatenate(axis=-1, name='concat1')([x1, x2, x3])
    x = Dropout(0.3, name='dropout1')(x)

    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', name='conv2a')(x)
    x1 = BatchNormalization(name='bn2a')(x1)
    x1 = MaxPool1D(pool_size=4, name='maxpool2a')(x1)

    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same', name='conv2b')(x)
    x2 = BatchNormalization(name='bn2b')(x2)
    x2 = MaxPool1D(pool_size=4, name='maxpool2b')(x2)

    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same', name='conv2c')(x)
    x3 = BatchNormalization(name='bn2c')(x3)
    x3 = MaxPool1D(pool_size=4, name='maxpool2c')(x3)

    x = Concatenate(axis=-1, name='concat2')([x1, x2, x3])
    x = Dropout(0.3, name='dropout2')(x)
    
    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPool1D(pool_size=2, name='maxpool3')(x)
    x = Dropout(0.3, name='dropout3')(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPool1D(pool_size=2, name='maxpool4')(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = MaxPool1D(pool_size=2, name='maxpool5')(x)
    x2 = Flatten(name='flatten')(x)

    x1 = Bidirectional(LSTM(32, return_sequences=True, name='lstm1'))(x)
    x1 = Bidirectional(LSTM(32, name='lstm2'))(x1)

    x2 = Dense(32, activation='relu', name='dense1')(x2)
    x = Concatenate(axis=-1, name='concat3')([x1, x2])
    x = Dropout(0.3, name='dropout4')(x)
    output = Dense(1, activation='relu', name='regressor')(x)
    return keras.Model(inp_sig, output)


def ft_ahi_model(inp_sig):
    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', name='conv1a')(inp_sig)
    x1 = BatchNormalization(name='bn1a')(x1)
    x1 = MaxPool1D(pool_size=4, name='maxpool1a')(x1)

    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same', name='conv1b')(inp_sig)
    x2 = BatchNormalization(name='bn1b')(x2)
    x2 = MaxPool1D(pool_size=4, name='maxpool1b')(x2)

    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same', name='conv1c')(inp_sig)
    x3 = BatchNormalization(name='bn1c')(x3)
    x3 = MaxPool1D(pool_size=4, name='maxpool1c')(x3)

    x = Concatenate(axis=-1, name='concat1')([x1, x2, x3])
    x = Dropout(0.3, name='dropout1')(x)

    x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', name='conv2a')(x)
    x1 = BatchNormalization(name='bn2a')(x1)
    x1 = MaxPool1D(pool_size=4, name='maxpool2a')(x1)

    x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same', name='conv2b')(x)
    x2 = BatchNormalization(name='bn2b')(x2)
    x2 = MaxPool1D(pool_size=4, name='maxpool2b')(x2)

    x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same', name='conv2c')(x)
    x3 = BatchNormalization(name='bn2c')(x3)
    x3 = MaxPool1D(pool_size=4, name='maxpool2c')(x3)

    x = Concatenate(axis=-1, name='concat2')([x1, x2, x3])
    x = Dropout(0.3, name='dropout2')(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPool1D(pool_size=2, name='maxpool3')(x)
    x = Dropout(0.3, name='dropout3')(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPool1D(pool_size=2, name='maxpool4')(x)

    x = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = MaxPool1D(pool_size=2, name='maxpool5')(x)
    x2 = Flatten(name='flatten')(x)

    x1 = Bidirectional(LSTM(32, return_sequences=True, name='lstm1'))(x)
    x1 = Bidirectional(LSTM(32, name='lstm2'))(x1)

    x2 = Dense(32, activation='relu', name='dense1')(x2)
    x = Concatenate(axis=-1, name='concat3')([x1, x2])
    x = Dropout(0.3, name='dropout4')(x)
    output = Dense(1, activation='relu', name='ft_regressor')(x)
    return keras.Model(inp_sig, output)
