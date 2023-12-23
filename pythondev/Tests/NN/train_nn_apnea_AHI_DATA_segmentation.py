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
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration, count_apneas_in_chunk, getApneaSegments, getWakeSegments, create_AHI_training_data
from Tests.NN.create_apnea_count_AHI_data_segmentation import get_apnea_labels, create_AHI_segmentation_training_data
import matplotlib.colors as mcolors
import tensorflow as tf
import datetime
import scipy.signal as sp
from train_U_net import Unet
db = DB()

from focal_loss import SparseCategoricalFocalLoss, BinaryFocalLoss
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
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')

    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')

    return parser.parse_args()


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

    all_data = {}
    training_data = {}
    test_data = {}
    data_files = fnmatch.filter(os.listdir(load_path), '*_X.*')
    label_files = fnmatch.filter(os.listdir(load_path), '*_y.*')
    print(data_files)
    print(label_files)

    rejected_sessions = [8734]

    for i, fn in enumerate(data_files):
        sess = int(fn[0:4])
        if sess in rejected_sessions:
            continue
        if int(sess) not in all_data.keys():
            all_data[sess] = {}
        if args.mode == 'train':
            all_data[sess]['X'] = np.load(os.path.join(load_path, fn), allow_pickle=True)

    for i, fn in enumerate(label_files):
        sess = int(fn[0:4])
        if sess not in all_data.keys():
            continue
        if args.mode == 'train':
            all_data[sess]['y'] = np.load(os.path.join(load_path, fn), allow_pickle=True)


    #k-fold
    stats = {0:[], 1:[], 2:[-1]}
    res_dict = {}
    cms = {}

    benchmark_setups = [8705, 8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735]
    non_benchmark_setups = [8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]
    AHI_incidence = {'low': [0, 0],
                     'mild': [0, 0],
                     'moderate': [0, 0],
                     'severe': [0, 0]}
    AHI_incidence_with_zero = {'none': [0,0],
                     'low': [0, 0],
                     'mild': [0, 0],
                     'moderate': [0, 0],
                     'severe': [0, 0]}
    all_gt = []
    all_pred = []

    all_AHI_gt = []
    all_AHI_pred = []

    fig2, bx = plt.subplots(2, gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    average_errors = []
    accuracies = []
    for k_fold_test_session in all_data.keys():
        fn = str(k_fold_test_session) + '_preds_3.png'

        if os.path.isfile(os.path.join(args.save_path, fn)) :
            print(k_fold_test_session, "already processed. skipping")
            continue
        print("::::::::::::::::::::", k_fold_test_session, "::::::::::::::::::::")
        json_fn = '/model/' + str(k_fold_test_session) + '_model.json'
        weights_fn = '/model/' + str(k_fold_test_session) + '_model.hdf5'
        if args.mode == 'train':
            test_sessions = [k_fold_test_session]

            main_session = db.session_from_setup(k_fold_test_session)
            radar_setups = db.setups_by_session(main_session)

            train_sessions = [s for s in all_data.keys() if s not in test_sessions]# and s in benchmark_setups]
            val_sessions = [s for s in all_data.keys() if s not in test_sessions]# and s not in benchmark_setups]

            print("train", train_sessions)
            print("test", test_sessions)

            for s in train_sessions:
                training_data[s] = {}
                training_data[s]['X'] = all_data[s]['X']
                training_data[s]['y'] = all_data[s]['y']
            for ts in test_sessions:
                test_data[ts] = {}
                test_data[ts]['X'] = all_data[ts]['X']
                test_data[ts]['y'] = all_data[ts]['y']


            data_test = []
            labels_test = []

            for s in test_sessions:
                try:
                    data_test.append(all_data[s]['X'])
                    labels_test.append(all_data[s]['y'])
                except:
                    print("session", s, "not ok")
                    continue

            print("train_sessions", train_sessions)
            print("val_sessions", val_sessions)
            print("test_session", test_sessions)
            data_test= np.vstack(data_test)
            labels_test = np.vstack(labels_test)

            epochs = 500
            batch_size = 64
            samples_per_epoch = 320

            input_sig_shape = data_test[0].shape
            if len(data_test[0].shape) == 1:
                input_sig_shape = (data_test[0].shape[0], 1)

            inp_sig = Input(shape=input_sig_shape)

            model = Unet(input_sig_shape)#Model(inp_sig, output)

            if not os.path.isdir(train_path + '/checkpoints/'):
                os.makedirs(train_path + '/checkpoints/')

            model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

            checkpointer = ModelCheckpoint(filepath=train_path + '/checkpoints/model.hdf5', verbose=1, save_best_only=True)
            log_dir = train_path +  "/" + str(k_fold_test_session)+"_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model_json = model.to_json()
            with open(train_path + "/checkpoints/model.json", "w") as json_file:
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
                    print(k_fold_test_session, "successfully loaded model from storage")
                except:
                    print(k_fold_test_session, "model not found")
                    failed_load = True

            if not args.reload or failed_load:
                #model.fit(my_generator(data_train, labels_train, batch_size),
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

                with open(train_path + json_fn, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(train_path + weights_fn)
                print("Saved model to disk")
                continue
        # test the NNf
        print(k_fold_test_session," TEST ----------------------------------------")

        json_file = open(train_path + json_fn, 'r')
        print("loading", train_path + json_fn)
        model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(model_json)
        # load weights into new model
        print("loading", train_path + weights_fn)
        model.load_weights(train_path + weights_fn)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

        respiration, fs_new = getSetupRespiration(k_fold_test_session)
        time_chunk = model.input_shape[1]
        #time_chunk = fs_new * chunk_size_in_minutes * 60
        step = fs_new * 5 * 60
        min_seg_size = 30.0 * fs_new

        # fig, ax = plt.subplots(3)
        apnea_segments = getApneaSegments(k_fold_test_session, respiration, fs_new)

        fig, ax = plt.subplots(3,gridspec_kw={"wspace": 0.2, "hspace": 0.5})
        for s in apnea_segments:
            c = 'red' if s[3] != 2.0 else 'green'
            ax[0].axvspan(s[0], s[1], color=c)

        apnea_ref_class = get_apnea_labels(k_fold_test_session)
        data_test, labels_test = create_AHI_segmentation_training_data(respiration=respiration, apnea_ref=apnea_ref_class,
                                                     time_chunk=time_chunk, step=time_chunk,
                                                     scale=args.scale, fs=fs_new)

        res_dict[k_fold_test_session] = []
        preds = model.predict(data_test)
        preds_argmax = np.argmax(preds, axis=2)
        #preds10 = np.repeat(preds_argmax, fs_new)

        for i in range(preds.shape[0]):
            plt.figure()
            plt.plot(preds_argmax[i,:], label='gt', linewidth=1)

            plt.plot(labels_test[i,:], label='prediction', linewidth=0.5)
            #preds = preds.reshape([len(preds)])
            plt.legend()
            fn = str(k_fold_test_session) + '_'+str(i)+'_unet_preds.png'
            plt.savefig(os.path.join(args.save_path, fn), dpi=300)
            plt.close()

        lt = labels_test.reshape(labels_test.shape[0]*labels_test.shape[1])
        pa = preds_argmax.reshape(preds_argmax.shape[0]*preds_argmax.shape[1])
        ax[1].plot(lt, label='gt', linewidth=0.5)
        ax[1].legend()
        ax[2].plot(pa, label='prediction', linewidth=0.5)
        #preds = preds.reshape([len(preds)])
        ax[2].legend()
        fn = str(k_fold_test_session) + '_unet_preds.png'
        plt.savefig(os.path.join(args.save_path, fn), dpi=300)

        continue
        err = np.abs(preds - labels_test)

        preds_long = np.zeros(len(respiration))
        labels_long = np.zeros(len(respiration))
        counter = 0
        for i in range(time_chunk, len(respiration), step):
            preds_long[i:i - step] = preds[counter]
            labels_long[i:i - time_chunk] = labels_test[counter]
            counter += 1
        mae = np.round(np.mean(err), 2)

        ax[0].plot(respiration, label='displacement')
        print("labels", labels_test)
        print("preds", preds)

        counter = 0
        #ax[0].plot(respiration, label='displacement')
        for i in range(time_chunk, len(respiration), step):
            #ax[0].axvspan(i - time_chunk, i, alpha=preds[int(counter)] / 50, color='blue', label=str(preds[int(counter)]))
            counter += 1
        UP = 1
        DOWN = int(time_chunk / step)
        filter = [0.15, 0.35, 0.35, 0.15]
        preds_lp = np.convolve(preds, filter, 'same')
        preds_lp = sp.resample_poly(preds_lp, UP, DOWN)
        if len(preds_lp) % 4 == 0:
            l = len(preds_lp)
        else:
            l = int(4 * (np.floor(len(preds_lp) / 4) + 1))
        preds_lp = np.array(list(preds_lp[:l]) + [0] * (l - len(preds_lp)))
        # plt.plot(labels_long, label='label')
        # plt.plot(preds_long, label='prediction')sp


        hours = [60 * 60 * fs_new * i for i in range(0, 11)]
        ahi_pred = np.abs(np.round(np.reshape(preds_lp, (-1, 4)).sum(axis=-1)))
        ahi_gt = np.zeros(len(hours) - 1)

        for i in range(len(hours) - 1):
            for a in apnea_segments:
                if a[1] > hours[i] and a[1] < hours[i + 1]:
                    ahi_gt[i] += 1
        len(respiration)

        print("AHI GT", ahi_gt)
        print("AHI PREDICTED (DB)", ahi_pred)

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
        ax[0].set_title(
            str(k_fold_test_session) + ' :: ' + str(corrects)+'/'+str(minlen)+' :: ' + str(acc)+'% :: '+str(mae)+' :: '+args.loss+'\nAHI PRED: ' + str(
                ahi_pred) + '\nAHI TRUE: ' +  str(ahi_gt))
        ax[0].title.set_size(10)
        ax[1].title.set_size(10)
        ax[2].title.set_size(10)
        fn = str(k_fold_test_session) + '_predictions_v4.png'
        plt.savefig(os.path.join(args.save_path, fn), dpi=300)
        #plt.show()
        plt.close(fig)

        print("Done...", k_fold_test_session, '------------------------------------')
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

        # col = list(mcolors.cnames.keys())
        # color_index = k_fold_test_session % len(col)
        # bx[0].scatter(labels_test, preds, s=3, color=col[color_index], label=str(k_fold_test_session))

        # bx[1].scatter(ahi_gt, ahi_pred, s=3, color=col[color_index], label=str(k_fold_test_session))
        # bx[0].title.set_text('Predictions vs labels (all)')
        # bx[1].title.set_text('Predictions vs labels (AHI)')
        # bx[0].title.set_size(10)
        # bx[1].title.set_size(10)
        # bx[0].set_xlabel('15 min. event count true', fontdict=font, loc='right')
        # bx[0].set_ylabel('15 min. event count predicted', fontdict=font)
        # bx[1].set_xlabel('AHI true', fontdict=font, loc='right')
        # bx[1].set_ylabel('AHI predicted', fontdict=font)
        # bx[0].legend(fontsize=5)
        # bx[1].legend(fontsize=5)
        #fn = str(k_fold_test_session) + '_regression.png'
        #plt.savefig(os.path.join(args.save_path, fn), dpi=1000)
    print(np.mean(accuracies), np.mean(average_errors))