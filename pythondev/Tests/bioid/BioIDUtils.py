import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization,Dropout,Input
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import traceback
import logging

folder = '//Neteera/Work/homes/eldar.hacohen/python_code/Vital-Signs-Tracking/pythondev/bioid/resources/nn_classifier'
model_path = os.path.join(folder, 'bioid_nn_weights.hdf5')
X_path = os.path.join(folder, 'X.npy')
X = np.load(X_path, allow_pickle=True)
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)


class NNTrainingParams:
    def __init__(self, epochs, batch_size=32, samples_per_epoch=320, sd=13):
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = sd


class NNModelData:
    def __init__(self, active_names_list=None, train_sessions=None, test_sessions=None,  nn_model=None):
        self.train_sessions = train_sessions
        self.test_sessions = test_sessions
        self.active_names = active_names_list
        self.model = nn_model
        self.name_2_class = None
        self.model_json_fn = None
        self.model_weights_fn = None

        if self.active_names is not None:
            self.name_2_class = {}
            for sidx, s in enumerate(self.active_names):
                self.name_2_class[s] = sidx


def create_bioid_model(num_classes, model_name, nn_input_shape):
    bioid_model = Sequential(name=model_name)
    bioid_model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=nn_input_shape))
    bioid_model.add(MaxPool2D(pool_size=(1, 2)))
    bioid_model.add(Conv2D(filters=64, kernel_size=(3, 1), activation='relu'))
    bioid_model.add(MaxPool2D(pool_size=(2, 1)))
    bioid_model.add(Conv2D(filters=32, kernel_size=(3, 1), activation='relu'))
    bioid_model.add(MaxPool2D(pool_size=(2, 1)))
    bioid_model.add(Conv2D(filters=64, kernel_size=(3, 1), activation='relu'))
    bioid_model.add(MaxPool2D(pool_size=(2, 1)))
    n_dims = 256
    bioid_model.add(Flatten())
    bioid_model.add(Dense(n_dims, activation='relu'))
    bioid_model.add(Dense(num_classes, activation='softmax'))

    #bioid_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    bioid_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(),
                     metrics=['accuracy'])
    return bioid_model


def create_heartbeat_data(n_heartbeats, model_data, names, idx):
    # orig loop for ref
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    for hb in range(n_heartbeats):  # iterate over individual heartbeats
        if names[hb] not in model_data.active_names:
            continue
        if idx[hb] in model_data.test_sessions:
            xtest.append(X[hb, :, :])
            ytest.append(model_data.name_2_class[names[hb]])

        elif idx[hb] in model_data.train_sessions:
            xtrain.append(X[hb, :, :])
            ytrain.append(model_data.name_2_class[names[hb]])

    ytest = np.array(ytest)
    ytrain = np.array(ytrain)

    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    return xtrain, ytrain, xtest ,ytest


def my_generator(samples, labels, n_c, batchsize):
    while 1:
        x_out = []
        y_out = []

        while len(x_out) < batchsize:
            cur_select = np.random.choice(samples.shape[0])
            x_out.append(samples[cur_select,:,:])
            y_out.append(labels[cur_select])

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)

        yield np.expand_dims(x_out, axis=3), keras.utils.to_categorical(y_out.reshape(y_out.shape[0], 1),
                                                                            num_classes=n_c,
                                                                            dtype='float32')


def remove_subject_from_nn(subject_name, metadata, base_model_data):
# 1. check if subject is in general data
    if subject_name not in metadata['names']:
        return []
    if subject_name not in base_model_data.active_names:
        return base_model_data

    subj_sessions = np.unique(metadata['name_2_sessions'][subject_name])
    # sessions = sessions[-10:]

    train_list = list(base_model_data.train_sessions)
    test_list = list(base_model_data.test_sessions)
    for session in subj_sessions:
        if session in base_model_data.train_sessions:
            train_list.remove(session)
        if session in base_model_data.test_sessions:
            test_list.remove(session)

    base_model_data.train_sessions = np.array(train_list)
    base_model_data.test_sessions = np.array(test_list)

    active_names_list = list(base_model_data.active_names)
    active_names_list.remove(subject_name)

    base_model_data.active_names = np.array(active_names_list)
    n_hb = len(metadata['idx'])

    base_model_data.name_2_class = {}

    for c_idx, subj_name in enumerate(base_model_data.active_names):
        base_model_data.name_2_class[subj_name] = c_idx

    n_classes = len(base_model_data.active_names)
    X_train_t, y_train_t, X_test_t, y_test_t = create_heartbeat_data(n_hb, base_model_data, metadata['names'], metadata['idx'])

    nn_input_shape = (X_train_t.shape[1],X_train_t.shape[2], 1)

    new_model = create_bioid_model(n_classes, "new_model_removed", nn_input_shape) #Sequential(name="new_model_added")

    try:
        print("loading weights from", base_model_data.model_weights_fn)
        new_model.load_weights(filepath=base_model_data.model_weights_fn, skip_mismatch=True, by_name=True)
        for layer_idx, layer in enumerate(new_model.layers):
            if layer_idx < len(new_model.layers) - 1:
                layer.trainable = False
    except Exception as e:
        logging.error(traceback.format_exc())


    training_params = NNTrainingParams(epochs=50)
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_train_t, y_train_t, test_size=0.2, random_state=training_params.seed)
    slash_idx = base_model_data.model_json_fn.rfind('/')
    s_path = base_model_data.model_json_fn[:slash_idx+1]

    base_model_data.model_json_fn = s_path+'bioid_nn_model.json'
    base_model_data.model_weights_fn = s_path+'bioid_nn_weights.hdf5'

    new_model.fit_generator(generator=my_generator(X_train_t, y_train_t, n_classes, training_params.batch_size),
                            steps_per_epoch=training_params.samples_per_epoch,
                            epochs=training_params.epochs,
                            verbose=1,
                            callbacks=[checkpointer],
                            validation_data=my_generator(X_val_t, y_val_t, n_classes, int(float(training_params.batch_size) / 6)),
                            class_weight=None,#weights,
                            workers=1,
                            shuffle=True,
                            validation_steps=10)
    new_model.save_weights(filepath=base_model_data.model_weights_fn)
    print("Saved added subject NN model")
    model_json_file = new_model.to_json()
    with open(base_model_data.model_json_fn, "w") as jf:
        jf.write(model_json_file)

    base_model_data.model = new_model
    return base_model_data


def add_subject_to_nn(subject_name, metadata, base_model_data):
    if subject_name not in metadata['session_2_name'].values():
        return []
    if subject_name in base_model_data.active_names:
        return base_model_data

    subj_sessions = np.unique(metadata['name_2_sessions'][subject_name])
    # sessions = sessions[-10:]
    split_point = int(0.8 * len(subj_sessions))

    list(base_model_data.train_sessions).append(list(subj_sessions[:split_point]))
    base_model_data.train_sessions = np.hstack([base_model_data.train_sessions, np.array(subj_sessions)[:split_point]])
    base_model_data.test_sessions = np.hstack([base_model_data.test_sessions, np.array(subj_sessions)[split_point:]])
    n_hb = len(metadata['idx'])

    base_model_data.active_names = base_model_data.active_names = np.hstack([base_model_data.active_names, subject_name])
    base_model_data.name_2_class[subject_name] = max(base_model_data.name_2_class.values())+1
    n_classes = len(base_model_data.active_names)
    X_train_t, y_train_t, X_test_t, y_test_t = create_heartbeat_data(n_hb, base_model_data, metadata['names'],
                                                                     metadata['idx'])

    nn_input_shape = (X_train_t.shape[1],X_train_t.shape[2], 1)

    new_model = create_bioid_model(n_classes, "new_model_added", nn_input_shape)

    try:
        print("loading weights from", base_model_data.model_weights_fn)
        new_model.load_weights(filepath=base_model_data.model_weights_fn, skip_mismatch=True, by_name=True)
        for layer_idx, layer in enumerate(new_model.layers):
            if layer_idx < len(new_model.layers) - 1:
                layer.trainable = False
    except Exception as e:
        logging.error(traceback.format_exc())


    training_params = NNTrainingParams(epochs=50)
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_train_t, y_train_t, test_size=0.2, random_state=training_params.seed)
    slash_idx = base_model_data.model_json_fn.rfind('/')
    s_path = base_model_data.model_json_fn[:slash_idx+1]

    base_model_data.model_json_fn = s_path+'bioid_nn_model.json'
    base_model_data.model_weights_fn = s_path+'bioid_nn_weights.hdf5'

    new_model.fit_generator(generator=my_generator(X_train_t, y_train_t, n_classes, training_params.batch_size),
                            steps_per_epoch=training_params.samples_per_epoch,
                            epochs=training_params.epochs,
                            verbose=1,
                            callbacks=[checkpointer],
                            validation_data=my_generator(X_val_t, y_val_t, n_classes, int(float(training_params.batch_size) / 6)),
                            class_weight=None,#weights,
                            workers=1,
                            shuffle=True,
                            validation_steps=10)
    new_model.save_weights(filepath=base_model_data.model_weights_fn)
    print("Saved added subject NN model")
    model_json_file = new_model.to_json()
    with open(base_model_data.model_json_fn, "w") as jf:
        jf.write(model_json_file)

    base_model_data.model = new_model
    return base_model_data
