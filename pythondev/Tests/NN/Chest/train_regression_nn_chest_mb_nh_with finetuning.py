#from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('//')
sys.path.append(conf_path + '/Tests/NN')
from Tests.vsms_db_api import *
import keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch
from sklearn import preprocessing
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration,  getApneaSegments
import matplotlib.colors as mcolors

from Tests.NN.create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels


db = DB()
col = list(mcolors.cnames.keys())
apnea_class = {'missing': -1,
                   'Normal Breathing': 0,
                   'normal': 0,
                   'Central Apnea': 1,
                   'Hypopnea': 2,
                   'Mixed Apnea': 3,
                   'Obstructive Apnea': 4,
                   'Noise': 5}

label_dict = {}
for k,v in apnea_class.items():
    label_dict[v] = k

counts = []
hypo_counts = []

def count_apneas_in_chunk(start_t, end_t, apnea_segments, hypopneas=True):
    count = 0
    hypo_count = 0
    if not apnea_segments:
        return None
    for a in apnea_segments:
        if a[0] in range(start_t, end_t) and a[1] in range(start_t, end_t):
            count += 1

            if a[3] == apnea_class['Hypopnea']:
                hypo_count += 1
    #print(start_t, "hypopneas", hypo_count, "total events", count)
    hypo_counts.append(hypo_count)
    return count


def create_AHI_training_data_no_wake(respiration, apnea_segments, wake_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []

    for i in range(time_chunk, len(respiration), step):
        wake_perc = 1.0 - (np.sum(wake_ref[int((i - time_chunk)/fs):int(i/fs)])/(time_chunk/fs))
        v = 1;
        if wake_perc > 0.75:
            v = v * 0
        else:
            v = v * 1

        seg = respiration[i - time_chunk:i]
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.robust_scale(seg))
        else:
            X.append(seg)
        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)
        y.append(num_apneas)
        valid.append(v)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        valid = np.stack(valid)
        print(X.shape, y.shape)
        return X, y, valid

    return X, y, valid



def create_AHI_training_data(respiration, apnea_segments, time_chunk, step, scale):
    X = []
    y = []
    valid = []
    #sleep_start = max(time_chunk, sleep_start) if sleep_start else time_chunk
    #sleep_end = min(len(respiration), sleep_end) if sleep_end else len(respiration)

    for i in range(time_chunk, len(respiration), step):

        seg = respiration[i - time_chunk:i]
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.robust_scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        y.append(num_apneas)
        valid.append(1)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        valid = np.stack(valid)
        return X, y, valid

    return X,y, valid
#

def my_generator(data, labels, batchsize):
    while 1:
        x_out = []
        y_out = []

        while len(x_out) < batchsize:
            cur_select = np.random.choice(len(data), 1)[0]
            x_out.append(data[cur_select])
            y_out.append(labels[cur_select])

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)
        yield x_out, y_out

def my_balanced_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        sessions = [s for s in data_dict.keys()]

        np.random.shuffle(sessions)

        for k in range(batchsize):
            i = np.random.choice(len(sessions), 1)[0]
            v = data_dict[sessions[i]]
            cur_select = np.random.choice(len(v['y']), 1)[0]
            x_out.append(v['X'][cur_select])
            y_out.append(v['y'][cur_select])
            #print(i, cur_select, len(x_out), len(y_out))

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)
        #print(x_out.shape, y_out.shape)
        yield x_out, y_out

def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the    cur_select = np.random.choice(len(v['y']), 1)[0]
                x_out.append(v['X'][cur_select])
                y_out.append(v['y'][cur_select])
                print(i, cur_select, len(x_out), len(y_out))
 types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-load_path_mb', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_nw', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')
    parser.add_argument('-loss', metavar='Loss', type=str, required=True, help='loss function')
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('--filter_wake_from_train', action='store_true', help='filter_wake_from_train')
    parser.add_argument('--disp', action='store_true', help='create display')
    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    train_path = args.save_path
    load_path_mb = args.load_path_mb
    load_path_nw = args.load_path_nw

    if args.scale:
        load_path_mb = os.path.join(args.load_path_mb, 'scaled')
        load_path_nw = os.path.join(args.load_path_nw, 'scaled')
    #     train_path = train_path + '_scaled'
    else:
        load_path_mb = os.path.join(args.load_path_mb, 'unscaled')
        load_path_nw = os.path.join(args.load_path_nw, 'unscaled')

    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)

    nw_setups = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735,
                  8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]
    mb_setups = [109816, 109817, 109818, 109838, 109858, 109860, 109866, 109870, 109871, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109896, 109897, 109901, 109903,
                 109906, 109910, 109918, 109922, 109924, 109925, 109926, 109928, 109937, 109945, 109946, 109957, 109958, 109965, 109966, 109971, 109972]

    all_setups = nw_setups + mb_setups
    stats = {0: [], 1: [], 2: [-1]}
    res_dict = {}
    cms = {}
    AHI_incidence = {'low': [0, 0],
                     'mild': [0, 0],
                     'moderate': [0, 0],
                     'severe': [0, 0]}
    AHI_incidence_with_zero = {'none': [0, 0],
                               'low': [0, 0],
                               'mild': [0, 0],
                               'moderate': [0, 0],
                               'severe': [0, 0]}
    all_gt = []
    all_pred = []
    all_hourly_gts = []
    all_hourly_preds = []
    all_AHI_gt = []
    all_AHI_pred = []
    all_AHI_gt_by_session = {}
    all_AHI_pred_by_session = {}
    mAHIs_gt = []
    mAHIs_pred = []
    errs_AHI = []
    err_by_session = {}

    completed_setups = []

    fig2, bx = plt.subplots(1, gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    #fig3, cx = plt.subplots(1, gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    average_errors = []
    accuracies = []
    res = {}
    completed_setups = []
    for k_fold_test_setup in all_setups:
        if k_fold_test_setup in completed_setups:
            continue
        all_data = {}
        training_data = {}

        data_files_mb = fnmatch.filter(os.listdir(load_path_mb), '*_X.*')
        data_files_nw = fnmatch.filter(os.listdir(load_path_nw), '*_X.*')
        refs_path_mb = load_path_mb#.replace("signals", "references")
        refs_path_nw = load_path_nw#.replace("signals", "references")
        label_files_mb = fnmatch.filter(os.listdir(refs_path_mb), '*_y.*')
        label_files_nw = fnmatch.filter(os.listdir(refs_path_nw), '*_y.*')
        valid_files_mb = fnmatch.filter(os.listdir(refs_path_mb), '*_valid*.*')
        valid_files_nw = fnmatch.filter(os.listdir(refs_path_nw), '*_valid*.*')
        data_files = data_files_mb + data_files_nw
        label_files = label_files_mb + label_files_nw
        valid_files = valid_files_mb + valid_files_nw

        for i, fn in enumerate(data_files):
            sess = int(fn[0:fn.find('_')])

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            load_path = []

            if fn in data_files_mb:
                load_path = load_path_mb
            elif fn in data_files_nw:
                load_path = load_path_nw
            else:
                continue
            all_data[sess]['X'] = np.load(os.path.join(load_path, fn), allow_pickle=True)
            sig = all_data[sess]['X'][0]

        for i, fn in enumerate(label_files):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            load_path = []

            if fn in label_files_mb:
                load_path = load_path_mb
            elif fn in label_files_nw:
                load_path = load_path_nw
            else:
                continue
            all_data[sess]['y'] = np.load(os.path.join(load_path, fn), allow_pickle=True)
        for i, fn in enumerate(valid_files):
            sess = int(fn[0:fn.find('_')])
            load_path = []

            if fn in valid_files_mb:
                load_path = load_path_mb
            elif fn in valid_files_nw:
                load_path = load_path_nw
            else:
                continue
            if sess not in all_data.keys():
                continue
            all_data[sess]['valid'] = np.load(os.path.join(load_path, fn), allow_pickle=True)

        json_fn = '/model/' + k_fold_test_setup + '_model.json'
        weights_fn = '/model/' + k_fold_test_setup + '_model.hdf5'
        if True:
            db.update_mysql_db(k_fold_test_setup)
            test_sessions = [k_fold_test_setup]
            subject_setups = db.setups_by_session(db.session_from_setup(k_fold_test_setup))
            train_sessions = [s for s in all_data.keys() if s not in subject_setups]  # if s not in radar_setups]# and s in benchmark_setups]

            print("train", train_sessions)
            print("test", test_sessions)

            for s in train_sessions:
                training_data[s] = {}
                training_data[s]['X'] = all_data[s]['X'][all_data[s]['valid'] == 1]
                training_data[s]['y'] = all_data[s]['y'][all_data[s]['valid'] == 1]

            epochs = 500
            batch_size = 128
            samples_per_epoch = 320

            input_sig_shape = all_data[s]['X'][0].shape
            if len(sig.shape) == 1:
                input_sig_shape = (sig.shape[0], 1)

            inp_sig = Input(shape=input_sig_shape)

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

            x1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)
            x1 = BatchNormalization()(x1)
            x1 = MaxPool1D(pool_size=4)(x1)

            x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(x)
            x2 = BatchNormalization()(x2)
            x2 = MaxPool1D(pool_size=4)(x2)

            x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(x)
            x3 = BatchNormalization()(x3)
            x3 = MaxPool1D(pool_size=4)(x3)

            x = Concatenate(axis=-1)([x1, x2, x3])
            x = Dropout(0.3)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            # x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=2)(x)
            x = Dropout(0.3)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            # x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=2)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=2)(x)
            x2 = Flatten()(x)

            x1 = Bidirectional(LSTM(32, return_sequences=True))(x)
            x1 = Bidirectional(LSTM(32))(x1)

            x2 = Dense(32, activation='relu')(x2)
            x = Concatenate(axis=-1)([x1, x2])
            # x = Dense(16, activation='relu')(x)
            x = Dropout(0.3)(x)
            output = Dense(1, activation='relu')(x)

            model = keras.Model(inp_sig, output)
           

            if not os.path.isdir(train_path + '/checkpoints/'):
                os.makedirs(train_path + '/checkpoints/')
            #model.summary()
            model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())

            checkpointer = ModelCheckpoint(
                filepath=train_path + '/checkpoints/'+ k_fold_test_setup + 'model.hdf5', verbose=1,
                save_best_only=True)
            #log_dir = train_path + "/" + str(k_fold_test_setup) + "_logs/fit/" + datetime.datetime.now().strftime(
            #    "%Y%m%d-%H%M%S")
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_json = model.to_json()
            with open(train_path + '/checkpoints/' + k_fold_test_setup+ 'model.json', "w") as json_file:
                json_file.write(model_json)
            failed_load = False

            if args.reload:
                try:
                    json_file = open(train_path + json_fn, 'r')
                    model_json = json_file.read()
                    json_file.close()
                    model = keras.models.model_from_json(model_json)
                    # load weights into new model
                    model.load_weights(train_path + weights_fn)
                    model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
                    print("successfully loaded model from storage")
                except:
                    print("model not found")
                    failed_load = True

            if not args.reload or failed_load:
                #model.fit(my_generator(data_train, labels_train, batch_size),
                model.fit(my_balanced_generator(training_data, batch_size),

                          #validation_data=my_generator(data_train, labels_train, batch_size),
                          validation_data=my_balanced_generator(training_data, batch_size),
                          steps_per_epoch=samples_per_epoch,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[checkpointer, EarlyStopping(patience=args.patience)],
                          class_weight=None,
                          workers=1,
                          shuffle=True,
                          validation_steps=10)

                if not os.path.isdir(train_path + '/model/'):
                    os.mkdir(train_path + '/model/')

                with open(train_path + json_fn, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(train_path + weights_fn)
                print("Saved model to disk")
                
                #continue
        # test the NN
        
        for setup_of_subject in subject_setups:
            print(setup_of_subject," TEST ----------------------------------------")

            completed_setups.append(setup_of_subject)
            # json_file = open(train_path + json_fn, 'r')
            # model_json = json_file.read()
            # json_file.close()
            # model = keras.models.model_from_json(model_json)
            # # load weights into new model
            # model.load_weights(train_path + weights_fn)
            # model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
    
            respiration, fs_new = getSetupRespiration(setup_of_subject)
            time_chunk = model.input_shape[1]
            #time_chunk = fs_new * chunk_size_in_minutes * 60
            step = fs_new * 5 * 60
            min_seg_size = 30.0 * fs_new
            apnea_segments = getApneaSegments(setup_of_subject, respiration, fs_new)
    
            fig, ax = plt.subplots(3,gridspec_kw={"wspace": 0.2, "hspace": 0.5})
            for s in apnea_segments:
                c = 'red' if s[3] != 2.0 else 'green'
                ax[0].axvspan(s[0], s[1], color=c)
    
    
            #data_test, labels_test = create_AHI_training_data(respiration=respiration, apnea_segments=apnea_segments, time_chunk=time_chunk, step=time_chunk, scale=args.scale, hypo=True)
            if args.filter_wake_from_train:
                ss_ref_class = get_sleep_stage_labels(setup_of_subject)
                data_test, labels_test, valid = create_AHI_training_data_no_wake(respiration=respiration, apnea_segments=apnea_segments,
                                                                     wake_ref=ss_ref_class, time_chunk=time_chunk,
                                                                     step=time_chunk, scale=args.scale, fs=fs_new)
            else:
                data_test, labels_test, valid = create_AHI_training_data(respiration=respiration, apnea_segments=apnea_segments,
                                                             time_chunk=time_chunk, step=time_chunk,
                                                             scale=args.scale)

            res_dict[setup_of_subject] = []
            preds = model.predict(data_test)
            preds = preds.reshape([len(preds)])
            err = np.abs(preds - labels_test)

            print(hypo_counts)
            perc = np.zeros_like(preds)
            for j in range(len(preds)):

                if labels_test[j] == 0:
                    perc[j] = 0
                else:
                    perc[j] = min(1, preds[j]/labels_test[j])
                print(j, hypo_counts[j], perc[j])
            print(perc)
            color_index = setup_of_subject % len(col)
            #cx.scatter(hypo_counts, perc, s=3, color=col[color_index])

            hypo_counts = []
            nsec = int(data_test.shape[1]/fs_new)

            if args.filter_wake_from_train:
                preds[valid == 0] = 0

            if len(preds) % 4 == 0:
                l = len(preds)
            else:
                l = int(4 * (np.floor(len(preds) / 4) + 1))
            preds = np.array(list(preds[:l]) + [0] * (l - len(preds)))
            labels_test = np.array(list(labels_test[:l]) + [0] * (l - len(labels_test)))
            ahi_pred = np.abs(np.round(np.reshape(preds, (-1, 4)).sum(axis=-1)))
            ahi_gt = np.abs(np.round(np.reshape(labels_test, (-1, 4)).sum(axis=-1)))

    
            print("AHI GT", ahi_gt)
            print("AHI PREDICTED (DB)", ahi_pred)
    
            all_hourly_gts.append(ahi_gt)
            all_hourly_preds.append(ahi_pred)
    
            AHI_dict = {'low': [0,5, 'green'],
                        'mild': [5,15, 'yellow'],
                        'moderate': [15,30, 'red'],
                        'severe': [30, max(max(ahi_gt), max(ahi_pred)), 'magenta']}
    
    
            for k,v in AHI_dict.items():
                #ax[2].axhline(y=v[1], color='gray', alpha=0.5)
                print(k)
                print(v)
                ax[2].axhspan(v[1], v[0], color=v[-1], alpha=0.3)
    
            ax[2].plot(ahi_gt, label='True', color='black', zorder=0)
            ax[2].plot(ahi_pred, label='Predicted', color='blue', zorder=0)
    
            minlen = min(len(labels_test), len(preds))
            labels_test = labels_test[:minlen]
            preds = preds[:minlen]
            all_pred.append(preds)
            all_gt.append(labels_test)
            minlen = min(len(ahi_pred), len(ahi_gt))
            ahi_gt = ahi_gt[:minlen]
            ahi_pred = ahi_pred[:minlen]
            all_AHI_pred.append(ahi_pred)
            all_AHI_gt.append(ahi_gt)
    
            AHI_correct = np.zeros(len(ahi_gt))
    
            mae = np.round(np.mean(np.abs(ahi_gt-ahi_pred)),2)
    
            for i in range(minlen):
                if ahi_gt[i] == 0 :
                    AHI_incidence_with_zero['none'][0] += 1
                if ahi_gt[i]  in range(1,5) :
                    AHI_incidence_with_zero['low'][0] += 1
                if ahi_gt[i] in range(AHI_dict['low'][0], AHI_dict['mild'][0]):
                    AHI_incidence['low'][0] +=1
                elif ahi_gt[i] in range(AHI_dict['mild'][0], AHI_dict['moderate'][0]):
                    AHI_incidence['mild'][0] +=1
                elif ahi_gt[i] in range(AHI_dict['moderate'][0], AHI_dict['severe'][0]):
                    AHI_incidence['moderate'][0] +=1
                elif ahi_gt[i] > AHI_dict['severe'][0]:
                   AHI_incidence['severe'][0] +=1
    
            for i in range(minlen):
                if ahi_gt[i] == 0 and ahi_pred[i] == 0:
                    AHI_incidence_with_zero['none'][1] += 1
                if ahi_gt[i] in range(1,5) and ahi_pred[i] in range(1,5):
                    AHI_incidence_with_zero['low'][1] += 1
                if ahi_gt[i] in range(AHI_dict['low'][0], AHI_dict['mild'][0]) and ahi_pred[i] in range(AHI_dict['low'][0], AHI_dict['mild'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['low'][1] +=1
                    ax[2].scatter(i, ahi_pred[i], color='k', marker='o', zorder=1)
                    ax[2].scatter(i, ahi_pred[i], color='g', marker='+', zorder=2)
                elif ahi_gt[i] in range(AHI_dict['mild'][0], AHI_dict['moderate'][0]) and ahi_pred[i] in range(AHI_dict['mild'][0], AHI_dict['moderate'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['mild'][1] +=1
                    ax[2].scatter(i, ahi_pred[i], color='k', marker='o', zorder=1)
                    ax[2].scatter(i, ahi_pred[i], color='g', marker='+', zorder=2)
                elif ahi_gt[i] in range(AHI_dict['moderate'][0], AHI_dict['severe'][0]) and ahi_pred[i] in range(AHI_dict['moderate'][0], AHI_dict['severe'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['moderate'][1] +=1
                    ax[2].scatter(i, ahi_pred[i], color='k', marker='o', zorder=1)
                    ax[2].scatter(i, ahi_pred[i], color='g', marker='+', zorder=2)
                elif ahi_gt[i] > AHI_dict['severe'][0] and ahi_pred[i] > AHI_dict['severe'][0]:
                    AHI_correct[i] = 1
                    AHI_incidence['severe'][1] +=1
                    ax[2].scatter(i, ahi_pred[i], color='k', marker='o', zorder=1)
                    ax[2].scatter(i, ahi_pred[i], color='g', marker='+', zorder=2)
                else:
                    ax[2].scatter(i, ahi_pred[i], color='k', marker='o', zorder=1)
                    ax[2].scatter(i, ahi_pred[i], color='r', marker='x', zorder=2)
    
            corrects = np.sum(AHI_correct)
            hours =len(ahi_gt)
            acc = np.round(100.0 * (corrects/minlen), 2)
    
            average_errors.append(mae)
            accuracies.append(acc)
    
            ax[1].plot(labels_test, label='True', color='black')
            ax[1].plot(preds, label='Predicted', color='blue')
            #ax[1].legend(fontsize=6)
    
            font = {'family': 'sans',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 6,
                    }
            ax[0].tick_params(axis='both', which='both', labelsize=6)
            ax[1].tick_params(axis='both', which='both', labelsize=6)
            ax[2].tick_params(axis='both', which='both', labelsize=6)
            #ax[3].tick_params(axis='both', which='both', labelsize=6)
    
            ax[2].set_xlabel('t(hours)', fontdict=font, loc='right')
            ax[2].set_ylabel('AHI', fontdict=font)
            #ax[0].set_xlabel('t(sec)', fontdict=font, loc='right')
            #ax[0].set_ylabel('displacement', fontdict=font)
            ax[0].set_xlabel('t(sec)', fontdict=font, loc='right')
            ax[0].set_ylabel('displacement', fontdict=font)
            ax[1].set_ylabel('apneas/segment', fontdict=font)
            ax[2].legend(fontsize=6)
            ax[2].title.set_text('AHI prediction accuracy by range')
            ax[1].title.set_text('AHI prediction by segment')
            m = min(len(ahi_gt), len(ahi_pred))
            ahi_ae = np.round(np.mean(np.abs(ahi_gt[:m] - ahi_pred[:m])), 2)
    
            m_AHI_gt = np.mean(ahi_gt)
            m_AHI_pred = np.mean(ahi_pred)
            err_AHI = np.round(np.abs(m_AHI_gt-m_AHI_pred),2)
            mAHIs_gt.append(m_AHI_gt)
            mAHIs_pred.append(m_AHI_pred)
            errs_AHI.append(err_AHI)
            err_by_session[setup_of_subject] = err_AHI

            res[setup_of_subject] = [m_AHI_gt, m_AHI_pred]

            all_AHI_pred_by_session[setup_of_subject] = m_AHI_pred
            all_AHI_gt_by_session[setup_of_subject] = m_AHI_gt
            color_index = setup_of_subject % len(col)
            bx.scatter(m_AHI_gt, m_AHI_pred, c='black', s=5)
            bx.scatter(m_AHI_gt, m_AHI_pred, c=col[color_index], label=str(setup_of_subject), s=3)

            bx.legend(fontsize='xx-small', loc='center left', bbox_to_anchor=(1, 0.5))


            ax[0].set_title(
                str(setup_of_subject) + ' :: ' + str(corrects)+'/'+str(minlen)+' :: ' + str(acc)+'% :: '+str(mae)+' :: '+args.loss+'\nGT M(AHI): '+str(m_AHI_gt)+' :: Pred M(AHI): '+str(m_AHI_pred)+' :: M(Error): '+str(err_AHI))
            ax[0].title.set_size(10)
            ax[1].title.set_size(10)
            ax[2].title.set_size(10)
            fn = 'a_'+str(setup_of_subject) + '_predictions_v5_filter_test2.png'
            #plt.savefig(os.path.join(args.save_path, fn), dpi=300)
            #plt.show()
            plt.close(fig)
    
            print("Done...", setup_of_subject, '------------------------------------')
            for k,v in AHI_incidence.items():
                print(k, v[0], v[1])
            print(AHI_incidence)
            print(AHI_incidence_with_zero)
    
    
            all_pred_array = np.hstack(all_pred)
            all_gt_array = np.hstack(all_gt)
            all_AHI_pred_array = np.hstack(all_AHI_pred)
            all_AHI_gt_array = np.hstack(all_AHI_gt)
    
            max_val = max(max(all_pred_array), max(all_gt_array))
            max_AHI_val = max(max(all_AHI_pred_array), max(all_AHI_gt_array))
    
    
            print("accuracies", np.mean(accuracies), "average_errors", np.mean(average_errors))
            print("mean errors AHIs", errs_AHI, np.mean(errs_AHI))
            print("mean GT AHIs", mAHIs_gt)
            print("mean predicted AHIs", mAHIs_pred)

            completed_setups.append(setup_of_subject)
            print(res)
    print("FINAL")
    print("accuracies", np.mean(accuracies), "average_errors", np.mean(average_errors))
    print("mean errors AHIs", errs_AHI, np.mean(errs_AHI))
    print("mean GT AHIs", mAHIs_gt)
    print("mean predicted AHIs", mAHIs_pred)
    print("mean AHI error by session", err_by_session)
    print("AHI gt by session", all_AHI_gt_by_session)
    print("AHI pred by session", all_AHI_pred_by_session)

    fn = str(setup_of_subject) + '_gt_vs_preds2_filter_test2.png'

    bx.plot([0, max_AHI_val], [5, 5], linewidth=0.5, c='gray')
    bx.plot([5, 5], [0, max_AHI_val], linewidth=0.5, c='gray')
    bx.plot([0, max_AHI_val], [15, 15], linewidth=0.5, c='gray')
    bx.plot([15,15], [0, max_AHI_val], linewidth=0.5, c='gray')
    bx.plot([0, max_AHI_val], [30,30], linewidth=0.5, c='gray')
    bx.plot([30,30], [0, max_AHI_val], linewidth=0.5, c='gray')
    bx.plot([0, max_AHI_val], [0, max_AHI_val], linewidth=0.5, c='gray')

    bx.set_title('Regression NN GT vs Predicions,  NWH setups')
    bx.set_xlabel('AHI GT')
    bx.set_ylabel('AHI PREDICTION')
    #plt.savefig(os.path.join(args.save_path, fn), dpi=300)

    #plt.close(fig2)
    #fn = 'hypos.png'
    #plt.savefig(os.path.join(args.save_path, fn), dpi=300)

    gt_class = np.zeros_like(mAHIs_gt)
    for i in range(len(mAHIs_gt)):
        if mAHIs_gt[i] < 5:
            gt_class[i] = 0
        elif mAHIs_gt[i] < 15:
            gt_class[i] = 1
        elif mAHIs_gt[i] < 30:
            gt_class[i] = 2
        else:
            gt_class[i] = 3

    pred_class = np.zeros_like(mAHIs_pred)
    for i in range(len(mAHIs_pred)):
        if mAHIs_pred[i] < 5:
            pred_class[i] = 0
        elif mAHIs_pred[i] < 15:
            pred_class[i] = 1
        elif mAHIs_pred[i] < 30:
            pred_class[i] = 2
        else:
            pred_class[i] = 3

    from sklearn.metrics import confusion_matrix

    print("GT", gt_class)
    print("P", pred_class)
    cm = confusion_matrix(gt_class, pred_class)
    print(cm)
    # report = {}
    # for cl in [0,1,2,3]:
    #     TP = 0
    #     FP = 0
    #     FN = 0
    #     TN = 0
    #
    #     for i in range(len(gt_class)):
    #         if gt_class[i] == cl:
    #             if pred_class[i] == cl:
    #                 TP += 1 #gt == cl & p == cl
    #             else:
    #                 FN += 1 #gt == cl & p != cl
    #         else:
    #             if pred_class[i] == cl:
    #                 FP += 1 #gt != cl & p == cl
    #             else:
    #                 TN += 1 #gt != cl & p != cl
    #     report[cl] = [TP/(TP+FN), TN/(TN+FP)]
    #
    #
    # print("sensitivity, specificity, session", report)

    all_hourly_gts = np.hstack(all_hourly_gts)
    all_hourly_preds = np.hstack(all_hourly_preds)
    print("15 min gt", len(all_hourly_gts), all_hourly_gts)
    print("15 min pred", len(all_hourly_preds), all_hourly_preds)
    cm_15 = confusion_matrix(all_hourly_gts, all_hourly_preds)
    print(cm_15)
    plt.figure()
    plt.scatter(all_hourly_gts, all_hourly_preds, s=3)
    plt.savefig(os.path.join(train_path, "scatter_15.png"))
    print(res)