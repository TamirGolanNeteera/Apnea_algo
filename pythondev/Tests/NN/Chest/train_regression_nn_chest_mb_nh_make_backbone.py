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
from keras.layers import Input
import numpy as np
import os
#import matplotlib.pyplot as plt
import fnmatch
from sklearn import preprocessing
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration,  getApneaSegments, getSetupRespirationCloudDB
import matplotlib.colors as mcolors
from Tests.NN.create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import create_AHI_regression_training_data_MB_chest_fs, load_apnea_ref_from_annotations, \
    apnea_class
from pylibneteera.ExtractPhaseFromCPXAPI import get_fs
from ahi_model import ahi_model

db = DB()
col = list(mcolors.cnames.keys())


label_dict = {}
for k,v in apnea_class.items():
    label_dict[v] = k

counts = []
hypo_counts = []

def count_apneas_in_chunk(start_t, end_t, apnea_segments):
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
            X.append(preprocessing.scale(seg))
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
            X.append(preprocessing.scale(seg))
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
    parser.add_argument('-load_path_mb_chest', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_mb_radar', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_nw_chest', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_nw_radar', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
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
    
    #handle chest folders
    load_path_mb_chest = args.load_path_mb_chest
    load_path_nw_chest = args.load_path_nw_chest

    if args.scale:
        load_path_mb_chest = os.path.join(args.load_path_mb_chest, 'scaled')
        load_path_nw_chest = os.path.join(args.load_path_nw_chest, 'scaled')
    else:
        load_path_mb_chest = os.path.join(args.load_path_mb_chest, 'unscaled')
        load_path_nw_chest = os.path.join(args.load_path_nw_chest, 'unscaled')

    #handle phase folders

    if args.scale:
        load_path_nw_radar = os.path.join(args.load_path_nw_radar, 'scaled')
        load_path_mb_radar = os.path.join(args.load_path_mb_radar, 'scaled')
        refs_path_mb_radar = os.path.join(args.load_path_mb_radar, 'scaled')
    else:
        load_path_nw_radar = os.path.join(args.load_path_nw_radar, 'unscaled')
        load_path_mb_radar = os.path.join(args.load_path_mb_radar, 'unscaled')
        refs_path_mb_radar = os.path.join(args.load_path_mb_radar, 'unscaled')
    #if args.filter_wake_from_train:
    #    load_path_nw_radar = os.path.join(load_path_nw_radar, 'filter_wake_from_train')


    refs_path_nw_radar = load_path_nw_radar.replace("signals", "references")
    
    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)

    nw_setups = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735,
                  8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]
    mb_setups = [109816, 109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897, 109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966, 110033, 110044, 110045, 110071, 110072]

    all_setups = nw_setups + mb_setups
    stats = {0: [], 1: [], 2: [-1]}

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

    ft_all_gt = []
    ft_all_pred = []
    ft_all_hourly_gts = []
    ft_all_hourly_preds = []
    ft_all_AHI_gt = []
    ft_all_AHI_pred = []
    ft_all_AHI_gt_by_session = {}
    ft_all_AHI_pred_by_session = {}
    ft_mAHIs_gt = []
    ft_mAHIs_pred = []
    ft_errs_AHI = []
    ft_err_by_session = {}

    completed_setups = []

    average_errors = []
    accuracies = []
    res = {}
    ft_res = {}
    completed_setups = []
    for k_fold_test_setup in all_setups:
        if k_fold_test_setup in completed_setups:
            continue
        all_data = {}
        chest_training_data = {}

        #make backbone
        data_files_mb_chest = fnmatch.filter(os.listdir(load_path_mb_chest), '*_X.*')
        data_files_nw_chest = fnmatch.filter(os.listdir(load_path_nw_chest), '*_X.*')
        refs_path_mb_chest = load_path_mb_chest  # .replace("signals", "references")
        refs_path_nw_chest = load_path_nw_chest  # .replace("signals", "references")
        label_files_mb_chest = fnmatch.filter(os.listdir(load_path_mb_chest), '*_y.*')
        label_files_nw_chest = fnmatch.filter(os.listdir(load_path_nw_chest), '*_y.*')
        valid_files_mb_chest = fnmatch.filter(os.listdir(load_path_mb_chest), '*_valid*.*')
        valid_files_nw_chest = fnmatch.filter(os.listdir(load_path_nw_chest), '*_valid*.*')
        data_files_chest = data_files_mb_chest + data_files_nw_chest
        label_files_chest = label_files_mb_chest + label_files_nw_chest
        valid_files = valid_files_mb_chest + valid_files_nw_chest

        for i, fn in enumerate(data_files_chest):
            sess = int(fn[0:fn.find('_')])

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            load_path = []

            if fn in data_files_mb_chest:
                load_path = load_path_mb_chest
            elif fn in data_files_nw_chest:
                load_path = load_path_nw_chest
            else:
                continue
            all_data[sess]['X'] = np.load(os.path.join(load_path, fn), allow_pickle=True)
            sig = all_data[sess]['X'][0]

        for i, fn in enumerate(label_files_chest):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            load_path = []

            if fn in label_files_mb_chest:
                load_path = load_path_mb_chest
            elif fn in label_files_nw_chest:
                load_path = load_path_nw_chest
            else:
                continue
            all_data[sess]['y'] = np.load(os.path.join(load_path, fn), allow_pickle=True)
        for i, fn in enumerate(valid_files):
            sess = int(fn[0:fn.find('_')])
            load_path = []

            if fn in valid_files_mb_chest:
                load_path = load_path_mb_chest
            elif fn in valid_files_nw_chest:
                load_path = load_path_nw_chest
            else:
                continue
            if sess not in all_data.keys():
                continue
            all_data[sess]['valid'] = np.load(os.path.join(load_path, fn), allow_pickle=True)

        json_fn = '/model/' + str(k_fold_test_setup) + '_backbone_model.json'
        weights_fn = '/model/' + str(k_fold_test_setup) + '_backbone_model.hdf5'

        #CHEST - here we select the specific data for this iteration/k_fold_test_setup
        db.update_mysql_db(k_fold_test_setup)
        test_sessions = [k_fold_test_setup]
        subject_setups = db.setups_by_session(db.session_from_setup(k_fold_test_setup))
        train_sessions = [s for s in all_data.keys() if
                          s not in subject_setups]  # if s not in radar_setups]# and s in benchmark_setups]

        print("train sessions chest backbone", train_sessions)
        print("test sessions", test_sessions)

        for s in train_sessions:
            chest_training_data[s] = {}
            chest_training_data[s]['X'] = all_data[s]['X'][all_data[s]['valid'] == 1]
            chest_training_data[s]['y'] = all_data[s]['y'][all_data[s]['valid'] == 1]

        epochs = 500
        batch_size = 64
        samples_per_epoch = 320

        input_sig_shape = all_data[s]['X'][0].shape
        if len(sig.shape) == 1:
            input_sig_shape = (sig.shape[0], 1)

        inp_sig = Input(shape=input_sig_shape)

        backbone_model = ahi_model(inp_sig)

        if not os.path.isdir(train_path + '/checkpoints/'):
            os.makedirs(train_path + '/checkpoints/')
        # model.summary()
        backbone_model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())

        checkpointer = ModelCheckpoint(
            filepath=train_path + '/checkpoints/' + str(k_fold_test_setup) + 'backbone_model.hdf5', verbose=1,
            save_best_only=True)

        backbone_model_json = backbone_model.to_json()
        with open(train_path + '/checkpoints/' + str(k_fold_test_setup) + 'backbone_model.json', "w") as json_file:
            json_file.write(backbone_model_json)
        failed_load = False

        if args.reload:
            try:
                json_file = open(train_path + json_fn, 'r')
                backbone_model_json = json_file.read()
                json_file.close()
                backbone_model = keras.models.model_from_json(backbone_model_json)
                # load weights into new model
                backbone_model.load_weights(train_path + weights_fn)
                backbone_model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
                print("successfully reloaded backbone_model from storage")
            except:
                print("backbone_model not found")
                failed_load = True

        if not args.reload or failed_load:
            # backbone_model.fit(my_generator(data_train, labels_train, batch_size),
            backbone_model.fit(my_balanced_generator(chest_training_data, batch_size),
                      validation_data=my_balanced_generator(chest_training_data, batch_size),
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
                json_file.write(backbone_model_json)
            # serialize weights to HDF5
            backbone_model.save_weights(train_path + weights_fn)
            print("Saved backbone chest backbone_model to disk")

        #now, read the RADAR data from storage
        ft_data = {}
        ft_training_data = {}
      
        #load_path_mb_radar = args.load_path_mb_radar
        #refs_path_mb_radar = args.load_path_mb_radar

        refs_path_nw_radar = load_path_nw_radar.replace("signals", "references")

        data_files_nw_radar = fnmatch.filter(os.listdir(load_path_nw_radar), '*_X.*')
        label_files_nw_radar = fnmatch.filter(os.listdir(refs_path_nw_radar), '*_y.*')
        print(data_files_nw_radar)
        print(label_files_nw_radar)
        data_files_mb_radar = fnmatch.filter(os.listdir(load_path_mb_radar), '*_X.*')
        label_files_mb_radar = fnmatch.filter(os.listdir(refs_path_mb_radar), '*_y.*')
        print(data_files_mb_radar)
        print(label_files_mb_radar)

        db.update_mysql_db(k_fold_test_setup)

        for i, fn in enumerate(data_files_nw_radar):
            sess = int(fn[0:fn.find('_')])
            if sess in [k_fold_test_setup] or sess == 8705:
                continue

            if int(sess) not in ft_data.keys():
                ft_data[sess] = {}
            ft_data[sess]['X'] = np.load(os.path.join(load_path_nw_radar, fn), allow_pickle=True)

        for i, fn in enumerate(label_files_nw_radar):
            sess = int(fn[0:fn.find('_')])
            if sess in [k_fold_test_setup] or sess == 8705:
                continue
            if sess not in ft_data.keys():
                continue
            ft_data[sess]['y'] = np.load(os.path.join(refs_path_nw_radar, fn), allow_pickle=True)

        for i, fn in enumerate(data_files_mb_radar):
            sess = int(fn[0:fn.find('_')])

            if int(sess) not in ft_data.keys():
                ft_data[sess] = {}
            ft_data[sess]['X'] = np.load(os.path.join(load_path_mb_radar, fn), allow_pickle=True)

        for i, fn in enumerate(label_files_mb_radar):
            sess = int(fn[0:fn.find('_')])

            if sess not in ft_data.keys():
                continue
            ft_data[sess]['y'] = np.load(os.path.join(refs_path_mb_radar, fn), allow_pickle=True)


        #select
        main_session = db.session_from_setup(k_fold_test_setup)
        radar_setups_all = db.setups_by_session(main_session) if k_fold_test_setup in nw_setups else [k_fold_test_setup]
        print(radar_setups_all)
        train_sessions = [s for s in ft_data.keys() if s not in radar_setups_all]

        test_sessions = [k_fold_test_setup]

        print("train", train_sessions)
        print("test", test_sessions)
        for tr in train_sessions:
            ft_training_data[tr] = {}
            ft_training_data[tr]['X'] = ft_data[tr]['X']
            ft_training_data[tr]['y'] = ft_data[tr]['y']
        print(ft_training_data.keys())

        #load model for transfer learning
        ft_model = ahi_model(inp_sig)

        try:
            json_file = open(train_path + json_fn, 'r')
            backbone_model_json = json_file.read()
            json_file.close()
            ft_model = keras.models.model_from_json(backbone_model_json)
            # load weights into new model
            ft_model.load_weights(filepath=train_path + weights_fn, skip_mismatch=True, by_name=True)

            print("successfully loaded backbone model from storage for finetuning")



            # for layer_idx, layer in enumerate(ft_model.layers):
            #
            #     if layer_idx < len(ft_model.layers) - 1:
            #         layer.trainable = False
                #print(layer, layer_idx, layer.trainable)
        except:
            print("backbone model not found")
            failed_load = True

        ft_model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
        #finetune the model on ft_data
        ft_model.fit(my_balanced_generator(ft_training_data, batch_size),
                      validation_data=my_balanced_generator(ft_training_data, batch_size),
                      steps_per_epoch=samples_per_epoch,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[checkpointer, EarlyStopping(patience=args.patience)],
                      class_weight=None,
                      workers=1,
                      shuffle=True,
                      validation_steps=10)

        json_fn = '/model/' + str(k_fold_test_setup) + '_finetuned_model.json'
        weights_fn = '/model/' + str(k_fold_test_setup) + '_finetuned_model.hdf5'
        ft_model_json = ft_model.to_json()
        if not os.path.isdir(train_path + '/model/'):
            os.mkdir(train_path + '/model/')

        with open(train_path + json_fn, "w") as json_file:
            json_file.write(ft_model_json)
        # serialize weights to HDF5
        ft_model.save_weights(train_path + weights_fn)
        print("Saved finetuned chest model to disk")

        # test the NN

        for setup_of_subject in subject_setups:
            print(setup_of_subject, " TEST ----------------------------------------")

            completed_setups.append(setup_of_subject)
            time_chunk = ft_model.input_shape[1]
            db.update_mysql_db(setup_of_subject)

            if db.mysql_db == 'neteera_cloud_mirror':
                apnea_reference = load_apnea_ref_from_annotations(setup_of_subject, db)
                print(apnea_reference)
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
                respiration , fs_new = getSetupRespirationCloudDB(setup_of_subject)
                step = fs_new * 5 * 60
                min_seg_size = 30.0 * fs_new

                fs_ref = get_fs(apnea_ref_class)
                data_test, labels_test, valid = create_AHI_regression_training_data_MB_chest_fs(respiration=respiration, apnea_ref=apnea_ref_class, time_chunk=time_chunk, step=step, scale=args.scale, fs=fs_new)
            else:
                respiration, fs_new = getSetupRespiration(setup_of_subject)
                step = fs_new * 5 * 60
                min_seg_size = 30.0 * fs_new
                apnea_segments = getApneaSegments(setup_of_subject, respiration, fs_new)

                if args.filter_wake_from_train:
                    ss_ref_class = get_sleep_stage_labels(setup_of_subject)
                    data_test, labels_test, valid = create_AHI_training_data_no_wake(respiration=respiration,
                                                                                     apnea_segments=apnea_segments,
                                                                                     wake_ref=ss_ref_class,
                                                                                     time_chunk=time_chunk,
                                                                                     step=time_chunk, scale=args.scale,
                                                                                     fs=fs_new)
                else:
                    data_test, labels_test, valid = create_AHI_training_data(respiration=respiration,
                                                                             apnea_segments=apnea_segments,
                                                                             time_chunk=time_chunk, step=time_chunk,
                                                                             scale=args.scale)

            preds = backbone_model.predict(data_test)
            ft_preds = ft_model.predict(data_test)


            preds = preds.reshape([len(preds)])
            err = np.abs(preds - labels_test)


            nsec = int(data_test.shape[1] / fs_new)

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

            print("Backbone AHI GT", ahi_gt)
            print("Backbone  AHI PREDICTED (DB)", ahi_pred)

            all_hourly_gts.append(ahi_gt)
            all_hourly_preds.append(ahi_pred)

            AHI_dict = {'low': [0, 5, 'green'],
                        'mild': [5, 15, 'yellow'],
                        'moderate': [15, 30, 'red'],
                        'severe': [30, max(max(ahi_gt), max(ahi_pred)), 'magenta']}

            for k, v in AHI_dict.items():
                print(k)
                print(v)

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

            mae = np.round(np.mean(np.abs(ahi_gt - ahi_pred)), 2)

            for i in range(minlen):
                if ahi_gt[i] == 0:
                    AHI_incidence_with_zero['none'][0] += 1
                if ahi_gt[i] in range(1, 5):
                    AHI_incidence_with_zero['low'][0] += 1
                if ahi_gt[i] in range(AHI_dict['low'][0], AHI_dict['mild'][0]):
                    AHI_incidence['low'][0] += 1
                elif ahi_gt[i] in range(AHI_dict['mild'][0], AHI_dict['moderate'][0]):
                    AHI_incidence['mild'][0] += 1
                elif ahi_gt[i] in range(AHI_dict['moderate'][0], AHI_dict['severe'][0]):
                    AHI_incidence['moderate'][0] += 1
                elif ahi_gt[i] > AHI_dict['severe'][0]:
                    AHI_incidence['severe'][0] += 1

            for i in range(minlen):
                if ahi_gt[i] == 0 and ahi_pred[i] == 0:
                    AHI_incidence_with_zero['none'][1] += 1
                if ahi_gt[i] in range(1, 5) and ahi_pred[i] in range(1, 5):
                    AHI_incidence_with_zero['low'][1] += 1
                if ahi_gt[i] in range(AHI_dict['low'][0], AHI_dict['mild'][0]) and ahi_pred[i] in range(
                        AHI_dict['low'][0], AHI_dict['mild'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['low'][1] += 1

                elif ahi_gt[i] in range(AHI_dict['mild'][0], AHI_dict['moderate'][0]) and ahi_pred[i] in range(
                        AHI_dict['mild'][0], AHI_dict['moderate'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['mild'][1] += 1

                elif ahi_gt[i] in range(AHI_dict['moderate'][0], AHI_dict['severe'][0]) and ahi_pred[i] in range(
                        AHI_dict['moderate'][0], AHI_dict['severe'][0]):
                    AHI_correct[i] = 1
                    AHI_incidence['moderate'][1] += 1

                elif ahi_gt[i] > AHI_dict['severe'][0] and ahi_pred[i] > AHI_dict['severe'][0]:
                    AHI_correct[i] = 1
                    AHI_incidence['severe'][1] += 1

            corrects = np.sum(AHI_correct)
            hours = len(ahi_gt)
            acc = np.round(100.0 * (corrects / minlen), 2)

            average_errors.append(mae)
            accuracies.append(acc)

            m = min(len(ahi_gt), len(ahi_pred))
            ahi_ae = np.round(np.mean(np.abs(ahi_gt[:m] - ahi_pred[:m])), 2)

            m_AHI_gt = np.mean(ahi_gt)
            m_AHI_pred = np.mean(ahi_pred)
            err_AHI = np.round(np.abs(m_AHI_gt - m_AHI_pred), 2)
            mAHIs_gt.append(m_AHI_gt)
            mAHIs_pred.append(m_AHI_pred)
            errs_AHI.append(err_AHI)
            err_by_session[setup_of_subject] = err_AHI

            res[setup_of_subject] = [m_AHI_gt, m_AHI_pred]

            all_AHI_pred_by_session[setup_of_subject] = m_AHI_pred
            all_AHI_gt_by_session[setup_of_subject] = m_AHI_gt

            print("Done...", setup_of_subject, '------------------------------------')
            # for k, v in AHI_incidence.items():
            #     print(k, v[0], v[1])
            # print(AHI_incidence)
            # print(AHI_incidence_with_zero)

            all_pred_array = np.hstack(all_pred)
            all_gt_array = np.hstack(all_gt)
            all_AHI_pred_array = np.hstack(all_AHI_pred)
            all_AHI_gt_array = np.hstack(all_AHI_gt)

            max_val = max(max(all_pred_array), max(all_gt_array))
            max_AHI_val = max(max(all_AHI_pred_array), max(all_AHI_gt_array))

            print("backbone accuracies", np.mean(accuracies), "average_errors", np.mean(average_errors))
            print("backbone mean errors AHIs", errs_AHI, np.mean(errs_AHI))
            print("backbone mean GT AHIs", mAHIs_gt)
            print("backbone mean predicted AHIs", mAHIs_pred)

            completed_setups.append(setup_of_subject)
            print("backbone res", res)

            #now, do the finetuning prediction

            ft_preds = ft_model.predict(data_test)
            ft_preds = ft_preds.reshape([len(ft_preds)])
            #err = np.abs(preds - labels_test)


            nsec = int(data_test.shape[1] / fs_new)

            if args.filter_wake_from_train:
                ft_preds[valid == 0] = 0

            if len(ft_preds) % 4 == 0:
                l = len(ft_preds)
            else:
                l = int(4 * (np.floor(len(ft_preds) / 4) + 1))
            ft_preds = np.array(list(ft_preds[:l]) + [0] * (l - len(ft_preds)))
            labels_test = np.array(list(labels_test[:l]) + [0] * (l - len(labels_test)))
            ft_ahi_pred = np.abs(np.round(np.reshape(ft_preds, (-1, 4)).sum(axis=-1)))
            ft_ahi_gt = np.abs(np.round(np.reshape(labels_test, (-1, 4)).sum(axis=-1)))

            print("FT AHI GT", ft_ahi_gt)
            print("FT AHI PREDICTED (DB)", ft_ahi_pred)

            ft_all_hourly_gts.append(ft_ahi_gt)
            ft_all_hourly_preds.append(ft_ahi_pred)


            ft_preds = ft_preds[:minlen]
            ft_all_pred.append(preds)
            ft_all_gt.append(labels_test)
            ft_ahi_gt = ft_ahi_gt[:minlen]
            ft_ahi_pred = ft_ahi_pred[:minlen]
            ft_all_AHI_pred.append(ft_ahi_pred)
            ft_all_AHI_gt.append(ft_ahi_gt)

            ft_AHI_correct = np.zeros(len(ft_ahi_gt))

            ft_m_AHI_gt = np.mean(ft_ahi_gt)
            ft_m_AHI_pred = np.mean(ft_ahi_pred)
            ft_err_AHI = np.round(np.abs(ft_m_AHI_gt - ft_m_AHI_pred), 2)
            ft_mAHIs_gt.append(ft_m_AHI_gt)
            ft_mAHIs_pred.append(ft_m_AHI_pred)
            ft_errs_AHI.append(ft_err_AHI)
            ft_err_by_session[setup_of_subject] = ft_err_AHI

            ft_res[setup_of_subject] = [ft_m_AHI_gt, ft_m_AHI_pred]

            ft_all_AHI_pred_by_session[setup_of_subject] = ft_m_AHI_pred
            ft_all_AHI_gt_by_session[setup_of_subject] = ft_m_AHI_gt

            print("FT Done...", setup_of_subject, '------------------------------------')
            for k, v in AHI_incidence.items():
                print(k, v[0], v[1])
            print(AHI_incidence)
            print(AHI_incidence_with_zero)

            all_pred_array = np.hstack(ft_all_pred)
            all_gt_array = np.hstack(ft_all_gt)
            all_AHI_pred_array = np.hstack(ft_all_AHI_pred)
            all_AHI_gt_array = np.hstack(ft_all_AHI_gt)

            max_val = max(max(all_pred_array), max(all_gt_array))
            max_AHI_val = max(max(all_AHI_pred_array), max(all_AHI_gt_array))

            print("FT accuracies", np.mean(accuracies), "average_errors", np.mean(average_errors))
            print("FT mean errors AHIs", errs_AHI, np.mean(errs_AHI))
            print("FT mean GT AHIs", mAHIs_gt)
            print("FT mean predicted AHIs", mAHIs_pred)

            completed_setups.append(setup_of_subject)
            print("FT res ", ft_res)

    print("FINAL")
    print("accuracies", np.mean(accuracies), "average_errors", np.mean(average_errors))
    print("mean errors AHIs", errs_AHI, np.mean(errs_AHI))
    print("mean GT AHIs", mAHIs_gt)
    print("mean predicted AHIs", mAHIs_pred)
    print("mean AHI error by session", err_by_session)
    print("AHI gt by session", all_AHI_gt_by_session)
    print("AHI pred by session", all_AHI_pred_by_session)


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


    all_hourly_gts = np.hstack(all_hourly_gts)
    all_hourly_preds = np.hstack(all_hourly_preds)
    print("15 min gt", len(all_hourly_gts), all_hourly_gts)
    print("15 min pred", len(all_hourly_preds), all_hourly_preds)
    cm_15 = confusion_matrix(all_hourly_gts, all_hourly_preds)
    print(cm_15)

    print("RES", res)
    print("FT RES", ft_res)