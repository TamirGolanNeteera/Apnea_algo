from __future__ import print_function
import argparse
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, concatenate
from keras.layers import Conv2D, MaxPool2D
import numpy as np
import os
from sklearn.model_selection import train_test_split


def my_generator(l_data, l_mean_data, batchsize):
    while 1:
        perc = 1 / len(l_data)
        x_mean_out = []
        x_out = []
        y_out = []

        for n in range(1, len(l_data) + 1):
            while len(x_out) < int(n * batchsize * perc):
                cur_select = np.random.choice(len(l_data[n - 1]), 1)[0]
                left = l_data[n - 1][cur_select][0].astype('float32')
                right = l_data[n - 1][cur_select][1].astype('float32')
                x_out.append(np.column_stack((left, right)))
                if len(l_mean_data):
                    x_mean_out.append(l_mean_data[n - 1][cur_select].astype('float32'))
                if n == 1:
                    y_out.append(0)
                else:
                    y_out.append(n - 2)

        x_out = np.asarray(x_out)
        if len(l_mean_data):
            x_mean_out = np.asarray(x_mean_out)
        y_out = np.asarray(y_out)
        if len(l_data_train) == 2:
            if len(l_mean_data):
                yield [np.expand_dims(x_out, axis=3), np.expand_dims(x_mean_out, axis=2)], np.expand_dims(y_out, axis=1)
            else:
                yield np.expand_dims(x_out, axis=3), np.expand_dims(y_out, axis=1)
        else:
            if len(l_mean_data):
                yield [np.expand_dims(x_out, axis=3), np.expand_dims(x_mean_out, axis=2)], \
                      keras.utils.to_categorical(y_out.reshape(y_out.shape[0], 1),
                                                 num_classes=len(l_data),
                                                 dtype='float32')
            else:
                yield np.expand_dims(x_out, axis=3), \
                      keras.utils.to_categorical(y_out.reshape(y_out.shape[0], 1),
                                                 num_classes=len(l_data),
                                                 dtype='float32')

def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-data_path', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')
    parser.add_argument('-layer_size', metavar='Length', type=int, required=True, help='length of the input layer')
    parser.add_argument('-window', metavar='window', type=int, required=True, help='time window of the signal')
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('-labels', metavar='Labels', nargs='+', type=str,
                        help='compute the following labels (default is all status)',
                        default=[a for a in ['motion', 'speaking', 'zrr', 'occupancy']])
    parser.add_argument('-label_dict_path', metavar='Location', type=str, required=False,
                        help='location of labels dictionary')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model')
    parser.add_argument('--mean_std', action='store_true', help='train also on mean and std of the signal and diff')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    l_size = args.layer_size
    if args.save_path:
        train_path = args.save_path
    else:
        train_path = args.data_path
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
    window = args.window
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)
    path_to_ref_data = os.path.join(args.data_path, 'y_train_' + str(int(l_size / window)) + 'hz.npy')
    path_to_sig_data = os.path.join(args.data_path, 'X_train_processed_' + str(int(l_size / window)) + 'hz.npy')
    y = np.load(path_to_ref_data, allow_pickle=True)
    X = np.load(path_to_sig_data, allow_pickle=True)

    if args.mean_std:
        path_to_mean_sig_data = os.path.join(args.data_path, 'X_mean_train' + '.npy')
        X_mean = np.load(path_to_mean_sig_data, allow_pickle=True)
    if args.label_dict_path:
        label_dict = np.load(args.label_dict_path, allow_pickle=True).item()
    else:
        label_dict = np.load(os.path.join(args.data_path, 'label_dict.npy'), allow_pickle=True).item()
    labels = [label_dict[x] for x in args.labels]
    l_data_train = []
    l_data_test = []
    if args.mean_std:
        l_mean_data_train = []
        l_mean_data_test = []

    def get_key(v):
        for k, value in label_dict.items():
            if v == value:
                return k
        return

    new_dict = {}

    for key, val in label_dict.items():
        if val == 0:
            if args.mean_std:
                X_train_l, X_test_l, X_mean_train_l, X_mean_test_l = \
                    train_test_split(X[np.count_nonzero(y == 0, axis=1) == window],
                                     X_mean[np.count_nonzero(y == 0, axis=1) == window],
                                     test_size=0.1,
                                     random_state=seed)
            else:
                X_train_l, X_test_l = train_test_split(X[np.count_nonzero(y == 0, axis=1) == window],
                                                       test_size=0.1,
                                                       random_state=seed)
        elif key == 'zrr':
            if args.mean_std:
                X_train_l, X_test_l, X_mean_train_l, X_mean_test_l = \
                    train_test_split(X[np.count_nonzero(y == val, axis=1) == window],
                                     X_mean[np.count_nonzero(y == val, axis=1) == window],
                                     test_size=0.1,
                                     random_state=seed)
            else:
                X_train_l, X_test_l = train_test_split(X[np.count_nonzero(y == val, axis=1) == window],
                                                       test_size=0.1,
                                                       random_state=seed)
        elif val in labels:
            if len(X[np.count_nonzero(y == val, axis=1) >= 2]) > 0:
                if args.mean_std:
                    X_train_l, X_test_l, X_mean_train_l, X_mean_test_l = \
                        train_test_split(X[np.count_nonzero(y == val, axis=1) >= 2],
                                         X_mean[np.count_nonzero(y == val, axis=1) >= 2],
                                         test_size=0.1,
                                         random_state=seed)
                else:
                    X_train_l, X_test_l, = train_test_split(X[np.count_nonzero(y == val, axis=1) >= 2],
                                                            test_size=0.1,
                                                            random_state=seed)
        else:
            continue
        l_data_train.append(X_train_l)
        l_data_test.append(X_test_l)
        if args.mean_std:
            l_mean_data_train.append(X_mean_train_l)
            l_mean_data_test.append(X_mean_test_l)

        new_dict[get_key(val)] = len(l_data_train) - 1
        print(get_key(val) + ': ' + str(l_data_train[len(l_data_train) - 1].shape))

    np.save(os.path.join(train_path, 'new_label_dict.npy'), new_dict)
    epochs = 500
    batch_size = 32
    samples_per_epoch = 320
    input_sig_shape = (l_size, 2, 1)
    if args.mean_std:
        input_mean_shape = (4, 1)

        inp_sig = Input(shape=input_sig_shape)
        conv_1 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(inp_sig)
        pool_1 = MaxPool2D(pool_size=(4, 1))(conv_1)
        conv_2 = Conv2D(filters=16, kernel_size=(5, 1), activation='relu')(pool_1)
        pool_2 = MaxPool2D(pool_size=(4, 1))(conv_2)
        conv_3 = Conv2D(filters=8, kernel_size=(3, 1), activation='relu')(pool_2)
        pool_3 = MaxPool2D(pool_size=(2, 1))(conv_3)
        conv_4 = Conv2D(filters=2, kernel_size=(3, 1), activation='relu')(pool_3)
        flat_sig = Flatten()(conv_4)
        dense_sig = Dense(40, activation='relu')(flat_sig)

        inp_mean = Input(shape=input_mean_shape)
        dense_mean = Dense(40, activation='relu')(inp_mean)
        flat_mean = Flatten()(dense_mean)
        dense_mean_2 = Dense(40, activation='relu')(flat_mean)
        merged = concatenate([dense_sig, dense_mean_2])
        if len(l_data_train) == 2:
            output = Dense(1, activation='sigmoid')(merged)
        else:
            output = Dense(len(l_data_train), activation='softmax')(merged)

        model = Model([inp_sig, inp_mean], output)
    else:
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(7, 1), activation='relu', input_shape=input_sig_shape))
        model.add(MaxPool2D(pool_size=(4, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(4, 1)))
        model.add(Conv2D(filters=16, kernel_size=(5, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(4, 1)))
        model.add(Conv2D(filters=8, kernel_size=(3, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 1)))
        model.add(Conv2D(filters=2, kernel_size=(3, 1), activation='relu'))
        model.add(Flatten())

        if len(l_data_train) == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(len(l_data_train), activation='softmax'))

    print(model.summary())

    if not os.path.isdir(train_path + '/checkpoints/'):
        os.makedirs(train_path + '/checkpoints/')
    else:
        if os.path.isfile(train_path + '/checkpoints/model.hdf5'):
            if not args.overwrite:
                print('loading checkpoint and continue training')
                json_file = open(train_path + '/checkpoints/model.json', 'r')
                model_json = json_file.read()
                json_file.close()
                model = keras.models.model_from_json(model_json)
                # load weights into new model
                model.load_weights(train_path + '/checkpoints/model.hdf5')

    if len(l_data_train) == 2:
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    else:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=train_path + '/checkpoints/model.hdf5', verbose=1, save_best_only=True)

    model_json = model.to_json()
    with open(train_path + "/checkpoints/model.json", "w") as json_file:
        json_file.write(model_json)

    if not args.mean_std:
        l_mean_data_train = []
        l_mean_data_test = []

    model.fit(my_generator(l_data_train, l_mean_data_train, batch_size),
              steps_per_epoch=samples_per_epoch,
              epochs=epochs,
              verbose=1,
              callbacks=[checkpointer, EarlyStopping(patience=50)],
              validation_data=my_generator(l_data_test, l_mean_data_test, int(float(batch_size) / 6)),
              class_weight=None,
              workers=1,
              shuffle=True,
              validation_steps=10)
    score = model.evaluate(my_generator(l_data_test, l_mean_data_test, int(float(batch_size) / 6)), verbose=0, steps=10)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if not os.path.isdir(train_path + '/model/'):
        os.mkdir(train_path + '/model/')

    with open(train_path + "/model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(train_path + "/model/model.hdf5")
    print("Saved model to disk")
