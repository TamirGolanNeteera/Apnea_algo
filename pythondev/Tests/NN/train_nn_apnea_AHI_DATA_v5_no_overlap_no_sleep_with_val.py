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

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import fnmatch
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration, count_apneas_in_chunk, getApneaSegments, getWakeSegments, create_AHI_training_data
import matplotlib.colors as mcolors
from Tests.NN.train_nn_apnea_AHI_DATA_segmentation_nsave import get_sleep_stage_labels


import datetime
import scipy.signal as sp
#import dbinterface
#db = dbinterface.DB()
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

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)
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

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)
        yield x_out, y_out
def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-load_path', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')
    parser.add_argument('-loss', metavar='Loss', type=str, required=True, help='loss function')
    parser.add_argument('-mode', metavar='running mode', type=str, required=True, help='running mode: train/test')
    #parser.add_argument('-layer_size', metavar='Length', type=int, required=True, help='length of the input layer')
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('--hypopneas', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('--disp', action='store_true', help='create display')
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

    all_setups = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735,
                  8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]

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

    average_errors = []
    accuracies = []
    res = {}
    for k_fold_test_setup in all_setups:
        if k_fold_test_setup in completed_setups:
            print(k_fold_test_setup, "already processed")
            continue
        all_data = {}
        training_data = {}
        val_data = {}

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

        for i, fn in enumerate(label_files):
            sess = int(fn[0:4])
            if sess in [k_fold_test_setup]:
                continue
            if sess not in all_data.keys():
                continue
            if args.mode == 'train':
                all_data[sess]['y'] = np.load(os.path.join(load_path, fn), allow_pickle=True)

        fn = str(k_fold_test_setup) + '_preds_3.png'

        if os.path.isfile(os.path.join(args.save_path, fn)) :
            print(k_fold_test_setup, "already processed. skipping")
            continue
        print("::::::::::::::::::::", k_fold_test_setup, "::::::::::::::::::::")
        json_fn = '/model/' + str(k_fold_test_setup) + '_model.json'
        weights_fn = '/model/' + str(k_fold_test_setup) + '_model.hdf5'
        if args.mode == 'train':
            main_session = db.session_from_setup(k_fold_test_setup)
            radar_setups_all = db.setups_by_session(main_session)
            radar_setups = [s for s in radar_setups_all if s in all_setups]

            train_sessions = [s for s in all_data.keys() if s not in radar_setups_all and s in all_setups]
            val_sessions = [s for s in all_data.keys() if s not in radar_setups and s not in all_setups]# and s not in benchmark_setups]
            
            test_sessions = [k_fold_test_setup]
            
            print("train", train_sessions)
            print("val", val_sessions)
            print("test", test_sessions)
            for tr in train_sessions:
                training_data[tr] = {}
                training_data[tr]['X'] = all_data[tr]['X']
                training_data[tr]['y'] = all_data[tr]['y']
            print(training_data.keys())

            for tr in val_sessions:
                val_data[tr] = {}
                val_data[tr]['X'] = all_data[tr]['X']
                val_data[tr]['y'] = all_data[tr]['y']
            print(training_data.keys())
            print("train_sessions", train_sessions)
            print("val_sessions", val_sessions)
            print("test_session", k_fold_test_setup)
            
            epochs = 500
            batch_size = 128
            samples_per_epoch = 320
            sig = all_data[tr]['X'][0]
            input_sig_shape = sig.shape
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
            x1 = BatchNormalization()(x)
            x1 = MaxPool1D(pool_size=4)(x)
            x2 = Conv1D(filters=8, kernel_size=5, activation='relu', padding='same')(x)
            x3 = BatchNormalization()(x)
            x2 = MaxPool1D(pool_size=4)(x)
            x3 = Conv1D(filters=8, kernel_size=7, activation='relu', padding='same')(x)
            x3 = BatchNormalization()(x)
            x3 = MaxPool1D(pool_size=4)(x)
            x = Concatenate(axis=-1)([x1, x2, x3])
            x = Dropout(0.3)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            # x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=4)(x)
            x = Dropout(0.3)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            # x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=4)(x)
            x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPool1D(pool_size=4)(x)
            x2 = Flatten()(x)

            x1 = Bidirectional(LSTM(64, return_sequences=True))(x)
            x1 = Bidirectional(LSTM(64))(x1)

            x2 = Dense(64, activation='relu')(x2)
            x = Concatenate(axis=-1)([x1, x2])
            # x = Dense(16, activation='relu')(x)
            x = Dropout(0.3)(x)
            output = Dense(1, activation='relu')(x)

            model = Model(inp_sig, output)
           

            if not os.path.isdir(train_path + '/checkpoints/'):
                os.makedirs(train_path + '/checkpoints/')

            model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())

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
                
                #continue
        # test the NN
        
        for setup_of_subject in radar_setups_all:
            print(setup_of_subject," TEST ----------------------------------------")
    
            json_file = open(train_path + json_fn, 'r')
            model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(model_json)
            # load weights into new model
            model.load_weights(train_path + weights_fn)
            model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
    
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
    
    
            data_test, labels_test = create_AHI_training_data(respiration=respiration, apnea_segments=apnea_segments, time_chunk=time_chunk, step=time_chunk, scale=args.scale, hypo=True)

            res_dict[setup_of_subject] = []
            preds = model.predict(data_test)
            preds = preds.reshape([len(preds)])
            err = np.abs(preds - labels_test)

            ss_ref_class = get_sleep_stage_labels(setup_of_subject)
            nsec = int(data_test.shape[1]/fs_new)
            preds_sleep_only = []
            labels_sleep_only = []
            for i in range(len(preds)):
                ss_chunk = ss_ref_class[i*nsec:min((i+1)*nsec, len(ss_ref_class))]
                print(i, np.unique(ss_chunk), i*nsec,min((i+1)*nsec, len(ss_ref_class)),len(ss_chunk[ss_chunk == 1]), nsec)
                if len(ss_chunk[ss_chunk == 1]) < 0.5 * nsec:
                    print('-')
                    continue

                preds_sleep_only.append(preds[i])
                labels_sleep_only.append(labels_test[i])

            print("len(preds_sleep_only)", len(preds_sleep_only), "len(labels_sleep_only)", len(labels_sleep_only))

            if len(preds_sleep_only) % 4 == 0:
                l = len(preds_sleep_only)
            else:
                l = int(4 * (np.floor(len(preds_sleep_only) / 4) + 1))
            preds = np.array(list(preds_sleep_only[:l]) + [0] * (l - len(preds_sleep_only)))
            labels_test = np.array(list(labels_sleep_only[:l]) + [0] * (l - len(labels_sleep_only)))

            nhours = int(len(preds)/4)
            hours = [60 * 60 * fs_new * i for i in range(0, nhours)]
            ahi_pred = np.abs(np.round(np.reshape(preds, (-1, 4)).sum(axis=-1)))
            ahi_gt = np.abs(np.round(np.reshape(labels_test, (-1, 4)).sum(axis=-1)))
            #ahi_gt = np.zeros(nhours - 1)
    
            # for i in range(nhours - 1):
            #     for a in apnea_segments:
            #         if a[1] > hours[i] and a[1] < hours[i + 1]:
            #             ahi_gt[i] += 1
            # len(respiration)
    
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

            col = list(mcolors.cnames.keys())

            color_index = setup_of_subject % len(col)
            bx.scatter(m_AHI_gt, m_AHI_pred, c='black', s=5)
            bx.scatter(m_AHI_gt, m_AHI_pred, c=col[color_index], label=str(setup_of_subject), s=3)

            bx.legend(fontsize='xx-small', loc='center left', bbox_to_anchor=(1, 0.5))


            ax[0].set_title(
                str(setup_of_subject) + ' :: ' + str(corrects)+'/'+str(minlen)+' :: ' + str(acc)+'% :: '+str(mae)+' :: '+args.loss+'\nGT M(AHI): '+str(m_AHI_gt)+' :: Pred M(AHI): '+str(m_AHI_pred)+' :: M(Error): '+str(err_AHI))
            ax[0].title.set_size(10)
            ax[1].title.set_size(10)
            ax[2].title.set_size(10)
            fn = 'a_'+str(setup_of_subject) + '_predictions_v5.png'
            plt.savefig(os.path.join(args.save_path, fn), dpi=300)
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

    fn = str(setup_of_subject) + '_gt_vs_preds2_.png'

    plt.plot([0, max_AHI_val], [5, 5], linewidth=0.5, c='gray')
    plt.plot([5, 5], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.plot([0, max_AHI_val], [15, 15], linewidth=0.5, c='gray')
    plt.plot([15,15], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.plot([0, max_AHI_val], [30,30], linewidth=0.5, c='gray')
    plt.plot([30,30], [0, max_AHI_val], linewidth=0.5, c='gray')
    plt.plot([0, max_AHI_val], [0, max_AHI_val], linewidth=0.5, c='gray')

    plt.title('Regression NN GT vs Predicions, 29 NWH setups')
    plt.xlabel('AHI GT')
    plt.ylabel('AHI PREDICTION')
    plt.savefig(os.path.join(args.save_path, fn), dpi=300)

    plt.close(fig2)

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
    report = {}
    for cl in [0,1,2,3]:
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in range(len(gt_class)):
            if gt_class[i] == cl:
                if pred_class[i] == cl:
                    TP += 1 #gt == cl & p == cl
                else:
                    FN += 1 #gt == cl & p != cl
            else:
                if pred_class[i] == cl:
                    FP += 1 #gt != cl & p == cl
                else:
                    TN += 1 #gt != cl & p != cl
        report[cl] = [TP/(TP+FN), TN/(TN+FP)]


    print("sensitivity, specificity, session", report)

    all_hourly_gts = np.hstack(all_hourly_gts)
    all_hourly_preds = np.hstack(all_hourly_preds)
    print("15 min gt", len(all_hourly_gts), all_hourly_gts)
    print("15 min pred", len(all_hourly_preds), all_hourly_preds)
    print(res)