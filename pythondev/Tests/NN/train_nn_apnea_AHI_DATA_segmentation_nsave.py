from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/')
sys.path.append(conf_path + '/Tests/NN')
from Tests.Plots.PlotRawDataRadarCPX import*
import keras
import copy
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration, getApneaSegments, getWakeSegments
from Tests.NN.create_apnea_count_AHI_data_segmentation_filter_wake_option import get_apnea_labels, get_sleep_stage_labels, create_AHI_segmentation_training_data, create_AHI_segmentation_training_data_no_wake
import matplotlib.colors as mcolors
import tensorflow as tf
import datetime
import scipy.signal as sp
from train_U_net import Unet
db = DB()


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



def my_generator(data, labels, batchsize):
    while 1:
        x_out = []
        y_out = []

        while len(x_out) < batchsize:
            cur_select = np.random.choice(len(data), 1)[0]
            x_out.append(data[cur_select])
            y_out.append(labels[cur_select])

        x_out = np.asarray(x_out, dtype=np.float32)
        y_out = np.asarray(y_out, dtype=np.float32)
        yield x_out, y_out

def my_balanced_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        sessions = [s for s in data_dict.keys()]
        np.random.shuffle(sessions)

        for k in sessions:
            for i in range(int(batchsize/len(sessions))):
                v = data_dict[k]
                cur_select = np.random.choice(len(v['y']), 1)[0]
                x_out.append(v['X'][cur_select])
                y_out.append(v['y'][cur_select])

        x_out = np.asarray(x_out, dtype=np.float32)
        y_out = np.asarray(y_out, dtype=np.float32)
        yield x_out, y_out
def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-load_path', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')

    parser.add_argument('-mode', metavar='running mode', type=str, required=True, help='running mode: train/test')

    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--filter_wake_from_train', action='store_true', help='filter_wake_from_train')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('--all', action='store_true', help='use all setups')

    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')

    return parser.parse_args()

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


def get_sleep_stage_labels(setup:int):
    ss_ref = None
    if setup in db.setup_nwh_benchmark():
        ss_ref = load_reference(setup, 'sleep_stages', db)
    else:
        session = db.session_from_setup(setup)
        setups = db.setups_by_session(session)
        for s in setups:
            if s in db.setup_nwh_benchmark():
                ss_ref = load_reference(s, 'sleep_stages', db)

    #print(sess, type(ss_ref))
    if ss_ref is None:
        return None

    if isinstance(ss_ref, pd.core.series.Series):
        ss_ref = ss_ref.to_numpy()


    ss_ref_class = np.zeros_like(ss_ref)

    for i in range(len(ss_ref)):
        if ss_ref[i] == 'W':
            ss_ref_class[i] = 0
        elif ss_ref[i] == None:
            ss_ref_class[i] = -1
        else:
            ss_ref_class[i] = 1
    return ss_ref_class

if __name__ == '__main__':

    args = get_args()

    if args.scale:
        load_path = os.path.join(args.load_path, 'scaled')
    else:
        load_path = os.path.join(args.load_path, 'unscaled')


    if args.save_path:
        train_path = args.save_path
    else:
        train_path = load_path

    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)



    all_setups =  [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735, 8710, 8705, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]

    stats = {0: [], 1: [], 2: [-1]}
    res= {}
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

    all_AHI_gt = []
    all_AHI_pred = []
    hourly_gt = []
    hourly_preds = []
    fig2, bx = plt.subplots(2, gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    average_errors = []
    accuracies = []

    for k_fold_test_setup in all_setups:

        all_data = {}
        training_data = {}

        data_files = fnmatch.filter(os.listdir(load_path), '*_X.*')
        label_files = fnmatch.filter(os.listdir(load_path), '*_y.*')
        print(data_files)
        print(label_files)

        for i, fn in enumerate(data_files):
            sess = int(fn[0:4])
            if sess in [k_fold_test_setup]:
                continue

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            if args.mode == 'train':
                all_data[sess]['X'] = np.load(os.path.join(load_path, fn), allow_pickle=True)
                sig = all_data[sess]['X'][0]

        for i, fn in enumerate(label_files):
            sess = int(fn[0:4])
            if sess in [k_fold_test_setup]:
                continue
            if sess not in all_data.keys():
                continue
            if args.mode == 'train':
                all_data[sess]['y'] = np.load(os.path.join(load_path, fn), allow_pickle=True)

        #fn = str(k_fold_test_setup) + '_preds_3.png'
        fn = 'all2_' + str(k_fold_test_setup) + '_unet_preds_no_wake.png'
        # if os.path.isfile(os.path.join(train_path, fn)) :
        #     print(k_fold_test_setup, "already processed. skipping")
        #     continue
        print("::::::::::::::::::::", k_fold_test_setup, "::::::::::::::::::::")
        json_fn = '/model/' + str(k_fold_test_setup) + '_model.json'
        weights_fn = '/model/' + str(k_fold_test_setup) + '_model.hdf5'
        if args.mode == 'train':
            main_session = db.session_from_setup(k_fold_test_setup)
            radar_setups = db.setups_by_session(main_session)
            if args.all:
                test_sessions = radar_setups
            else:
                test_sessions = [k_fold_test_setup]


            train_sessions = [s for s in all_data.keys() if s not in radar_setups]# and s in benchmark_setups]

            print("train", train_sessions)
            print("test", test_sessions)

            for s in train_sessions:
                training_data[s] = {}
                training_data[s]['X'] = all_data[s]['X']
                training_data[s]['y'] = all_data[s]['y']

            epochs = 500
            batch_size = 128
            samples_per_epoch = 320

            print(all_data.keys())

            input_sig_shape = sig.shape
            if len(sig.shape) == 1:
                input_sig_shape = (sig.shape[0], 1)

            inp_sig = Input(shape=input_sig_shape)

            model = Unet(input_sig_shape)#Model(inp_sig, output)

            if not os.path.isdir(train_path + '/checkpoints/'):
                os.makedirs(train_path + '/checkpoints/')

            model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())
            #model.compile(loss=SparseCategoricalFocalLoss(gamma=2), optimizer=keras.optimizers.Adam())

            checkpointer = ModelCheckpoint(filepath=train_path + '/checkpoints/'+str(k_fold_test_setup)+'_model.hdf5', verbose=1, save_best_only=True)
            log_dir = train_path +  "/" + str(k_fold_test_setup)+"_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_json = model.to_json()
            with open(train_path + '/checkpoints/'+str(k_fold_test_setup)+'model.json', "w") as json_file:
                json_file.write(model_json)

            failed_load = False

            if args.reload:
                try:
                    print("loading", json_file)
                    json_file = open(train_path + json_fn, 'r')
                    model_json = json_file.read()
                    json_file.close()
                    model = keras.models.model_from_json(model_json)
                    # load weights into new model
                    print("loading", train_path + weights_fn)
                    model.load_weights(train_path + weights_fn)
                    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())
                    #model.compile(loss=SparseCategoricalFocalLoss(gamma=2), optimizer=keras.optimizers.Adam())

                    print(k_fold_test_setup, "successfully loaded model from storage")
                except:
                    print(k_fold_test_setup, "model not found")
                    failed_load = True

            if not args.reload or failed_load:
                #model.fit(my_generator(data_train, labels_train, batch_size),
                #model.summary()
                model.fit(my_balanced_generator(training_data, batch_size),

                          #validation_data=my_generator(data_train, labels_train, batch_size),
                          validation_data=my_balanced_generator(training_data, batch_size),
                          steps_per_epoch=samples_per_epoch,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[tensorboard_callback, checkpointer, EarlyStopping(patience=args.patience)],
                          class_weight=None,
                          workers=1,
                          shuffle=True,
                          validation_steps=10)

                if not os.path.isdir(train_path + '/model/'):
                    os.mkdir(train_path + '/model/')

                with open(train_path + json_fn, "w") as jjson_file:
                    jjson_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(train_path + weights_fn)
                print("Saved model to disk")
                #continue
        # test the NNf
        for setup_of_subject in test_sessions:
            print(setup_of_subject," TEST ----------------------------------------")
            if setup_of_subject == 8705:
                print(setup_of_subject, "invalid")
                continue

            respiration, fs_new = getSetupRespiration(setup_of_subject)
            time_chunk = model.input_shape[1]
            #time_chunk = fs_new * chunk_size_in_minutes * 60
            #step = fs_new * 5 * 60
            min_seg_size = 30.0 * fs_new


            #create a 1 hz getApneaSegments
            apnea_segments = getApneaSegments(setup_of_subject, respiration, fs_new)

            fig, ax = plt.subplots(3,gridspec_kw={"wspace": 0.2, "hspace": 0.5})
            for s in apnea_segments:
                c = 'red' if s[3] != 2.0 else 'green'
                ax[0].axvspan(s[0], s[1], color=c, alpha=0.5)

            ss_ref_class = get_sleep_stage_labels(setup_of_subject)
            if len(ss_ref_class) == 0:
                print('ref bad')
                continue

            apnea_ref_class = get_apnea_labels(setup_of_subject)
            if len(apnea_ref_class) == 0:
                print('ref bad')
                continue

            wakedata = getWakeSegments(setup_of_subject)
            print(len(apnea_ref_class), wakedata[0])
            for s in wakedata[0]:
                ax[1].axvspan(s[0], s[1], color='yellow', alpha=0.3)
                ax[2].axvspan(s[0], s[1], color='yellow', alpha=0.3)

            if args.filter_wake_from_train:

                data_test, labels_test, valid = create_AHI_segmentation_training_data_no_wake(respiration=respiration, apnea_ref=apnea_ref_class,
                                                                     wake_ref=ss_ref_class, time_chunk=time_chunk,
                                                                     step=time_chunk, scale=args.scale, fs=fs_new)
            else:
                data_test, labels_test, valid = create_AHI_segmentation_training_data(respiration=respiration, apnea_ref=apnea_ref_class,
                                                             time_chunk=time_chunk, step=time_chunk,
                                                             scale=args.scale, fs=fs_new)



            preds = model.predict(data_test)
            preds_argmax = np.argmax(preds, axis=2)
            preds_max = np.max(preds, axis=2)
            #preds10 = np.repeat(preds_argmax, fs_new)
            ahi_gt_all = []
            ahi_pred_all = []

            for i in range(preds.shape[0]):
                # end_idx = min((i+1)*labels_test.shape[1], len(ss_ref_class))
                # ss_chunk = ss_ref_class[i*labels_test.shape[1]:end_idx]

                #print(setup_of_subject, i, np.unique(ss_chunk), i*labels_test.shape[1], end_idx, len(ss_chunk[ss_chunk == 1]))
                # if len(ss_chunk[ss_chunk == 1]) < 0.5 * labels_test.shape[1]:
                #     #print('-')
                #     continue

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



                # plt.figure()
                # # plt.plot(ss_chunk, label='sleep/wake', linewidth=3)
                # plt.plot(preds_argmax[i,:], label='prediction', linewidth=2)
                # plt.plot(preds_max[i, :], label='score', linewidth=1)
                # plt.plot(labels_test[i,:], label='gt', linewidth=1)
                #
                # #preds = preds.reshape([len(preds)])
                # plt.legend()
                # fn = str(setup_of_subject) + '_'+str(i)+'_unet_preds_no_wake.png'
                # plt.savefig(os.path.join(args.save_path, fn), dpi=300)
                #
                #
                # plt.close()

            print(ahi_gt_all)
            print(ahi_pred_all)

            hourly_gt.append(ahi_gt_all)
            hourly_preds.append(ahi_pred_all)


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

            lt = labels_test.reshape(labels_test.shape[0]*labels_test.shape[1])
            pa = preds_argmax.reshape(preds_argmax.shape[0]*preds_argmax.shape[1])
            ax[1].plot(lt, label='gt', linewidth=0.5, alpha=0.3)
            ax[1].legend()
            ax[2].plot(pa, label='prediction', linewidth=0.5,alpha=0.3)
            #preds = preds.reshape([len(preds)])
            ax[2].legend()

            #AHI_GT = np.round(np.mean(ahi_gt_all_extended), 2)
            #AHI_PRED = np.round(np.mean(ahi_pred_all_extended),2)

            plt.title(str(setup_of_subject) + ' AHI GT ' + str(AHI_GT) + ' AHI PRED ' + str(AHI_PRED))
            fn = 'all2_' + str(setup_of_subject) + '_unet_preds_no_wake.png'
            plt.savefig(os.path.join(train_path, fn), dpi=300)
            plt.show()
            plt.close()
            res[setup_of_subject] = [AHI_GT, AHI_PRED]
            print(res)

    plt.figure()
    col = list(mcolors.cnames.keys())

    max_AHI_val = 0
    for k,v in res.items():
        color_index = k % len(col)
        plt.scatter(v[0], v[1], c='black', s=5)
        plt.scatter(v[0], v[1], c=col[color_index], label=str(k), s=3)
        m = max(v)
        max_AHI_val = max(max_AHI_val, m)
    plt.legend(fontsize='xx-small', loc='center left', bbox_to_anchor=(1,0.5))
    plt.title('Unet GT vs Predicions, 29 NWH setups')
    plt.xlabel('AHI GT')
    plt.ylabel('AHI PREDICTION')
    fn = 'all_' + str(setup_of_subject) + '_unet_preds_vs_gt2.png'
    plt.plot([0, max_AHI_val], [5, 5], linewidth=0.5, c='gray')
    plt.plot([5, 5], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.plot([0, max_AHI_val], [15, 15], linewidth=0.5, c='gray')
    plt.plot([15, 15], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.plot([0, max_AHI_val], [30, 30], linewidth=0.5, c='gray')
    plt.plot([30, 30], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.savefig(os.path.join(train_path, fn), dpi=300)
    plt.close()
    gt = []
    preds = []

    for k,v in res.items():
        if v[0] < 5:
            gt.append(0)
        elif v[0]< 15:
            gt.append(1)
        elif v[0]< 30:
            gt.append(2)
        else:
            gt.append(3)

        if v[1] < 5:
            preds.append(0)
        elif v[1]< 15:
            preds.append(1)
        elif v[1]< 30:
            preds.append(2)
        else:
            preds.append(3)

    report_hourly = {}
    for cl in [0, 1, 2, 3]:
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in range(len(gt)):
            if gt[i] == cl:
                if preds[i] == cl:
                    TP += 1  # gt == cl & p == cl
                else:
                    FN += 1  # gt == cl & p != cl
            else:
                if preds[i] == cl:
                    FP += 1  # gt != cl & p == cl
                else:
                    TN += 1  # gt != cl & p != cl
        if TP + FN == 0:
            sensitivity = 0
        else:
            sensitivity = TP / (TP + FN)
        if TN + FP == 0:
            specificity = 0
        else:
            specificity = TN/(TN + FP)
        report_hourly[cl] = [sensitivity, specificity]

    print("4 class sensitivity, specificity, session", report_hourly)
    from sklearn.metrics import confusion_matrix

    cm_m = confusion_matrix(gt, preds)
    print("cm, session")
    print(cm_m)
    report_hourly = {}

    print('3 class ***************************')

    gt = []
    preds = []

    gt_plot = []
    pred_plot = []

    for k, v in res.items():
        gt_plot.append(v[0])
        pred_plot.append(v[1])
        if v[0] < 5:
            gt.append(0)
        elif v[0] < 30:
            gt.append(1)
        else:
            gt.append(2)

        if v[1] < 5:
            preds.append(0)
        elif v[1] < 30:
            preds.append(1)
        else:
            preds.append(2)


    report_hourly = {}
    for cl in [0, 1, 2]:
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in range(len(gt)):
            if gt[i] == cl:
                if preds[i] == cl:
                    TP += 1  # gt == cl & p == cl
                else:
                    FN += 1  # gt == cl & p != cl
            else:
                if preds[i] == cl:
                    FP += 1  # gt != cl & p == cl
                else:
                    TN += 1  # gt != cl & p != cl
        if TP + FN == 0:
            sensitivity = 0
        else:
            sensitivity = TP / (TP + FN)
        if TN + FP == 0:
            specificity = 0
        else:
            specificity = TN / (TN + FP)
        report_hourly[cl] = [sensitivity, specificity]

    print("3 class, sensitivity, specificity, session", report_hourly)



    hourly_gt = np.hstack(hourly_gt)
    hourly_preds = np.hstack(hourly_preds)

    mx = max([max(hourly_gt), max(hourly_preds)])
    plt.figure()
    plt.scatter(gt_plot, pred_plot, s=3, label='mean')
    plt.scatter(hourly_gt, hourly_preds, s=3, label='hourly')

    plt.title('AHI estimation by U-Net')
    plt.xlabel("AHI Reference")
    plt.ylabel("AHI Predicted")
    plt.legend()


    print("15 min gt", hourly_gt)


