#from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/')
sys.path.append(conf_path + '/Tests/NN')
from Tests.vsms_db_api import *
import keras
import sys
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
import pandas as pd
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration,  getApneaSegments, getSetupRespirationCloudDB
import matplotlib.colors as mcolors
from Tests.Utils.LoadingAPI import load_reference, load_phase
from Tests.NN.create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels
from Tests.NN.create_apnea_count_AHI_data import MB_HQ, create_AHI_training_data_no_wake_no_empty
#from Tests.NN.create_apnea_count_AHI_data_regression_MB_phase_filter_empty import load_apnea_ref_from_annotations, apnea_class, get_empty_seconds_mb
from Tests.NN.create_apnea_count_AHI_data_SEGMENTATION_MB_phase_filter_empty import create_AHI_segmentation_training_data_MB_phase, load_apnea_ref_from_annotations, apnea_class, get_empty_seconds_mb
from Tests.NN.create_apnea_count_AHI_data_SEGMENTATION_NW_phase_filter_empty import create_AHI_segmentation_NW_no_wake_no_empty, get_apnea_labels
from Tests.NN.create_apnea_count_AHI_data_NW_phase_filter_empty import get_empty_seconds_nw
from pylibneteera.ExtractPhaseFromCPXAPI import get_fs
from train_U_net import Unet
db = DB()
col = list(mcolors.cnames.keys())


label_dict = {}
for k,v in apnea_class.items():
    label_dict[v] = k

counts = []
hypo_counts = []

def count_apneas_in_chunk(start_t, end_t, apnea_segments, hypopneas=True):
    count = 0

    if not apnea_segments:
        return None
    for a in apnea_segments:
        if a[0] in range(start_t, end_t) and a[1] in range(start_t, end_t):
            count += 1
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

        x_out = np.asarray(x_out, dtype=np.float32)
        y_out = np.asarray(y_out, dtype=np.float32)
        #print(x_out.shape, y_out.shape)
        yield x_out, y_out

def compute_apnea_segments_from_labels(labels):
    apnea_segments = []
    apnea_diff = np.diff(labels, prepend=0, append=0)

    apnea_changes = np.where(apnea_diff)[0]

    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]

    for a_idx, start_idx in enumerate(apnea_idx):
        end_idx = apnea_end_idx[a_idx]
        apnea_segments.append([start_idx, end_idx, apnea_duration])

    return apnea_segments


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-load_path', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_mb', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')
    parser.add_argument('-loss', metavar='Loss', type=str, required=True, help='loss function')
    #parser.add_argument('-mode', metavar='running mode', type=str, required=True, help='running mode: train/test')
    #parser.add_argument('-layer_size', metavar='Length', type=int, required=True, help='length of the input layer')
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('--filter_wake_from_train', action='store_true', help='filter_wake_from_train')
    #parser.add_argument('--disp', action='store_true', help='create display')
    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    train_path = args.save_path
    load_path_nw = args.load_path
    load_path_mb = args.load_path_mb
    refs_path_mb = args.load_path_mb

    if args.scale:
        load_path_nw = os.path.join(load_path_nw, 'scaled')
        load_path_mb = os.path.join(load_path_mb, 'scaled')
        refs_path_mb = os.path.join(refs_path_mb, 'scaled')
    else:
        load_path_nw = os.path.join(load_path_nw, 'unscaled')
        load_path_mb = os.path.join(load_path_mb, 'unscaled')
        refs_path_mb = os.path.join(refs_path_mb, 'unscaled')

    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    with open(os.path.join(train_path, 'command.txt'), 'w') as f:
        f.write(sys.argv[0])
        f.write(str(sys.argv))


    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)

    nw_setups = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735,
                  8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]

    mb_setups = MB_HQ#[109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897,
 #109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966, 110033, 110044, 110045, 110071, 110072, 110190, 110191]

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
    segment_data_gt = {}
    segment_data_preds = {}
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
    all_setups =  mb_setups + nw_setups
    print(len(all_setups), "setups")
    for k_fold_test_setup in all_setups:
        if k_fold_test_setup in completed_setups:
            print(k_fold_test_setup, "already processed")
            continue
        all_data = {}
        training_data = {}
        val_data = {}


        refs_path_nw = load_path_nw.replace("signals", "references")

        data_files_nw = fnmatch.filter(os.listdir(load_path_nw), '*_X.*')
        label_files_nw = fnmatch.filter(os.listdir(refs_path_nw), '*_y.*')
        valid_files_nw = fnmatch.filter(os.listdir(refs_path_nw), '*_valid.*')
        #print(data_files_nw)
        #print(label_files_nw)
        data_files_mb = fnmatch.filter(os.listdir(load_path_mb), '*_X.*')
        label_files_mb = fnmatch.filter(os.listdir(refs_path_mb), '*_y.*')
        valid_files_mb = fnmatch.filter(os.listdir(refs_path_mb), '*_valid.*')
        #print(data_files_mb)
        #print(label_files_mb)

        db.update_mysql_db(k_fold_test_setup)

        for i, fn in enumerate(data_files_nw):
            sess = int(fn[0:fn.find('_')])
            if sess in [k_fold_test_setup] or sess == 8705:
                continue

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            all_data[sess]['X'] = np.load(os.path.join(load_path_nw, fn), allow_pickle=True)
            # if np.isnan(all_data[sess]['X']).any():
            #     print(sess, "X contains nan")

        for i, fn in enumerate(label_files_nw):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            all_data[sess]['y'] = np.load(os.path.join(refs_path_nw, fn), allow_pickle=True)

            # if np.isnan(all_data[sess]['y']).any():
            #     print(sess, "y contains nan")
        print(valid_files_nw)
        for i, fn in enumerate(valid_files_nw):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            all_data[sess]['valid'] = np.load(os.path.join(refs_path_nw, fn), allow_pickle=True)

            # if np.isnan(all_data[sess]['valid']).any():
            #     print(sess, "valid contains nan")

        for i, fn in enumerate(data_files_mb):
            sess = int(fn[0:fn.find('_')])

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            all_data[sess]['X'] = np.load(os.path.join(load_path_mb, fn), allow_pickle=True)
            # if np.isnan(all_data[sess]['X']).any():
            #     print(sess, "X contains nan")

        for i, fn in enumerate(label_files_mb):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            all_data[sess]['y'] = np.load(os.path.join(refs_path_mb, fn), allow_pickle=True)

            #(sess, all_data[sess]['y'].shape)
            # if (all_data[sess]['y'] == None).any() or np.isnan(all_data[sess]['y']).any():
            #     print(sess, "y contains nan")

        for i, fn in enumerate(valid_files_mb):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            all_data[sess]['valid'] = np.load(os.path.join(refs_path_mb, fn), allow_pickle=True)

            # if np.isnan(all_data[sess]['valid']).any():
            #     print(sess, "valid contains nan")

        print("::::::::::::::::::::", k_fold_test_setup, "::::::::::::::::::::")
        json_fn = '/model/' + str(k_fold_test_setup) + '_model.json'
        weights_fn = '/model/' + str(k_fold_test_setup) + '_model.hdf5'
        if True:
            main_session = db.session_from_setup(k_fold_test_setup)
            radar_setups_all = db.setups_by_session(main_session) if k_fold_test_setup in nw_setups else [k_fold_test_setup]

            train_sessions = [s for s in all_data.keys() if s not in radar_setups_all and s in all_setups]
            val_sessions = [s for s in all_data.keys() if s not in radar_setups_all and s not in all_setups]# and s not in benchmark_setups]
            
            test_sessions = [k_fold_test_setup]
            
           # print("train", train_sessions)
           # print("val", val_sessions)
           # print("test", test_sessions)
            for tr in train_sessions:
                training_data[tr] = {}
                training_data[tr]['X'] = all_data[tr]['X'][all_data[tr]['valid'] == 1]
                training_data[tr]['y'] = all_data[tr]['y'][all_data[tr]['valid'] == 1]

            #print(training_data.keys())

            epochs = 500
            batch_size = 128
            samples_per_epoch = 320
            sig = all_data[tr]['X'][0]
            input_sig_shape = sig.shape
            if len(sig.shape) == 1:
                input_sig_shape = (sig.shape[0], 1)

            inp_sig = Input(shape=input_sig_shape)

            model = Unet(input_sig_shape)  # Model(inp_sig, output)

            if not os.path.isdir(train_path + '/checkpoints/'):
                os.makedirs(train_path + '/checkpoints/')

            model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

            checkpointer = ModelCheckpoint(
                filepath=train_path + '/checkpoints/' + str(k_fold_test_setup) + '_model.hdf5', verbose=1,
                save_best_only=True)
            log_dir = train_path + "/" + str(k_fold_test_setup) + "_logs/fit/" + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_json = model.to_json()
            with open(train_path + '/checkpoints/' + str(k_fold_test_setup) + 'model.json', "w") as json_file:
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
                    print(k_fold_test_setup, "successfully loaded model from storage")
                except:
                    print(k_fold_test_setup, "model not found")
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
        
        for setup_of_subject in radar_setups_all:
            print(setup_of_subject," TEST ----------------------------------------")
    
            # json_file = open(train_path + json_fn, 'r')
            # model_json = json_file.read()
            # json_file.close()
            # model = keras.models.model_from_json(model_json)
            # # load weights into new model
            # model.load_weights(train_path + weights_fn)
            # model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
            time_chunk = model.input_shape[1]
            if setup_of_subject in nw_setups:
                respiration, fs_new = getSetupRespiration(setup_of_subject)
                step = time_chunk
                min_seg_size = 30.0 * fs_new
                apnea_segments = getApneaSegments(setup_of_subject, respiration, fs_new)

                ss_ref_class = get_sleep_stage_labels(setup_of_subject)
                empty_ref_nw = get_empty_seconds_nw(sess)
                apnea_reference = load_apnea_ref_from_annotations(setup_of_subject, db)
                # respiration, fs_new = getSetupRespirationCloudDB(sess)
                if apnea_reference is None:
                    print("no reference for setup", sess)
                    continue
                if isinstance(apnea_reference, pd.core.series.Series):
                    apnea_reference = apnea_reference.to_numpy()

                apnea_ref_class = np.zeros(len(apnea_reference))

                for i in range(len(apnea_reference)):
                    # print(i, apnea_reference[i], apnea_reference[i] in apnea_class.keys())
                    if apnea_reference[i] not in apnea_class.keys():
                        apnea_ref_class[i] = -1
                    else:
                        apnea_ref_class[i] = int(apnea_class[apnea_reference[i]])

                print(np.unique(apnea_ref_class))
                if len(apnea_ref_class) == 0:
                    print('ref bad')
                    continue
                print(sess, np.unique(apnea_ref_class))

                data_test, labels_test, valid = create_AHI_segmentation_NW_no_wake_no_empty(respiration=respiration, apnea_ref=apnea_ref_class, wake_ref=ss_ref_class, empty_ref=empty_ref_nw, time_chunk=time_chunk, step=step, scale=args.scale, fs=fs_new)

            else:
                apnea_reference = load_apnea_ref_from_annotations(setup_of_subject, db)
                #print(apnea_reference)
                if apnea_reference is None:
                    print("no reference for setup", setup_of_subject)
                    continue

                print(np.unique(apnea_reference))
                apnea_ref_class = np.zeros(len(apnea_reference))
                for i in range(len(apnea_reference)):
                    if apnea_reference[i] not in apnea_class.keys():
                        apnea_ref_class[i] = -1
                    else:
                        apnea_ref_class[i] = int(apnea_class[apnea_reference[i]])

                print(np.unique(apnea_ref_class))
                respiration , fs_new, bins = getSetupRespirationCloudDB(setup_of_subject)
                step = time_chunk
                min_seg_size = 30.0 * fs_new

                fs_ref = get_fs(apnea_ref_class)
                empty_ref_mb = get_empty_seconds_mb(setup_of_subject)
                print("respiration", len(respiration), "empty_sec_mb", len(empty_ref_mb), "bins", len(bins))

                if len(bins) != len(empty_ref_mb):
                    print("respiration", len(respiration), "empty_sec_mb", len(empty_ref_mb), "bins", len(bins))
                if len(empty_ref_mb) < len(bins):
                    empty_ref_mb = np.pad(empty_ref_mb, [0, len(bins) - len(empty_ref_mb)], mode='constant',
                                          constant_values=(0))
                elif len(empty_ref_mb) > len(bins):
                    empty_ref_mb = empty_ref_mb[:len(bins)]
                if len(bins) != len(empty_ref_mb):
                    print("***")
                empty_bins = [1 if f == -1 else 0 for f in bins]

                data_test, labels_test, valid = create_AHI_segmentation_training_data_MB_phase(respiration=respiration, apnea_ref=apnea_ref_class, empty_seconds=empty_ref_mb, time_chunk=time_chunk, step=step, scale=args.scale, fs=fs_new)

            res_dict[setup_of_subject] = []
            preds = model.predict(data_test)
            preds_argmax = np.argmax(preds, axis=2)
            preds_max = np.max(preds, axis=2)
            #preds10 = np.repeat(preds_argmax, fs_new)
            ahi_gt_all = []
            ahi_pred_all = []

            for i in range(preds.shape[0]):


                if len(np.unique(labels_test[i,:])) == 1:
                    ahi_gt_all.append(0)
                else:
                    segments_gt = compute_apnea_segments_from_labels(labels_test[i,:])
                    AHI_GT = len(segments_gt)
                    ahi_gt_all.append(AHI_GT)

                if len(np.unique(preds_argmax[i,:])) == 1:
                    ahi_pred_all.append(0)
                else:
                    segments_pred = compute_apnea_segments_from_labels(preds_argmax[i,:])
                    AHI_PRED = len(segments_pred)
                    ahi_pred_all.append(AHI_PRED)

            print("GT",ahi_gt_all)
            print("PREDS",ahi_pred_all)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_gt.npy'), ahi_gt_all)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_valid.npy'), valid)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_pred.npy'), ahi_pred_all)
            if len(ahi_gt_all) % 4 == 0:
                l = len(ahi_gt_all)
            else:
                l = int(4 * (np.floor(len(ahi_gt_all) / 4) + 1))
            ahi_gt_all_extended = np.array(list(ahi_gt_all[:l]) + [0] * (l - len(ahi_gt_all)))

            if len(ahi_pred_all) % 4 == 0:
                l = len(ahi_pred_all)
            else:
                l = int(4 * (np.floor(len(ahi_pred_all) / 4) + 1))
            ahi_pred_all_extended = np.array(list(ahi_pred_all[:l]) + [0] * (l - len(ahi_pred_all)))

            ahi_pred_all_extended = np.abs(np.round(np.reshape(ahi_pred_all_extended, (-1, 4)).sum(axis=-1)))
            ahi_gt_all_extended = np.abs(np.round(np.reshape(ahi_gt_all_extended, (-1, 4)).sum(axis=-1)))

            AHI_GT = np.round(np.mean(ahi_gt_all_extended), 2)
            AHI_PRED = np.round(np.mean(ahi_pred_all_extended), 2)

            lt = labels_test.reshape(labels_test.shape[0] * labels_test.shape[1])
            pa = preds_argmax.reshape(preds_argmax.shape[0] * preds_argmax.shape[1])

            print("Done...", setup_of_subject, '------------------------------------')
            for k,v in AHI_incidence.items():
                print(k, v[0], v[1])
            print(AHI_incidence)
            print(AHI_incidence_with_zero)
    


    print(res)