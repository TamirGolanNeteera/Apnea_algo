#from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/')
sys.path.append(conf_path + '/Tests/NN')
import os
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import keras
# from tensorflow import keras
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

import os
import fnmatch
import matplotlib.colors as mcolors
from Tests.NN.create_apnea_count_AHI_data import MB_HQ, NW_HQ#create_AHI_training_data_no_wake_no_empty
from Tests.vsms_db_api import DB
db = DB()
col = list(mcolors.cnames.keys())
import numpy as np
import scipy as sci
import pandas as pd
import random
import time as time
from Tests.NN.nn_models import small_vgg_model, large_resnet_model_late_batchnorm_1D,  small_resnet_model_late_batchnorm
from keras.utils.vis_utils import plot_model
def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')


#TODO:
#make generator that loads setup feature vector, selects random location and crops sample vector


def my_balanced_augmenting_generator(data_dict,  batchsize):
    while 1:
        x_out = []
        y_out = []

        sessions = [s for s in data_dict.keys()]

        np.random.shuffle(sessions)

        for k in range(batchsize):
            i = np.random.choice(len(sessions), 1)[0]
            v = data_dict[sessions[i]]

            sample = np.load
            x_label = 'X'
            y_label = 'y'

            cur_select = np.random.choice(len(v[y_label]), 1)[0]
            y = v[y_label][cur_select]

            if np.isnan(v[x_label][cur_select]).any():
                continue

            if y > 5 and np.random.rand() > 0.5:
                x_o = np.zeros_like(v[x_label][cur_select])


                half = int(len(x_o)/2)

                x_o[:half] = v[x_label][cur_select][half:]
                x_o[half:] = v[x_label][cur_select][:half]
                x_out.append(x_o)

            else:
                if y == 0 and np.random.rand() > 0.5:
                    continue
                else:
                    x_out.append(v[x_label][cur_select])

            y_out.append(v[y_label][cur_select])
            #print(i, cur_select, len(x_out), len(y_out))

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)
        #print(x_out.shape, y_out.shape)
        yield x_out, y_out

def my_weighted_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        category = random.choices(list(data_dict['data'].keys()), weights=data_dict['weights'], k=batchsize)
        #print("category", category)
        for k in range(batchsize):

            data_of_category = data_dict['data'][category[k]]

            cur_select = np.random.choice(len(data_of_category['y']), 1)[0]

            if np.random.rand() > 0.5:
                x_o = np.zeros_like(data_of_category['X'][cur_select])

                half = int(len(x_o)/2)

                x_o[:half] = data_of_category['X'][cur_select][half:]
                x_o[half:] = data_of_category['X'][cur_select][:half]
                x_out.append(x_o)
            else:
                x_out.append(data_of_category['X'][cur_select])
            y_out.append(data_of_category['y'][cur_select])

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)

        yield x_out, y_out


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-gpu', metavar='gpu', type=int, required=False, help='gpu device id')
    parser.add_argument('-load_path_nw', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-load_path_mb', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    #parser.add_argument('-load_path_ch', metavar='LoadPath', type=str, required=True, help='Path from which to load file')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output model')
    parser.add_argument('-loss', metavar='Loss', type=str, required=True, help='loss function')
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('-bs', metavar='seed', type=int, required=False, help='Batch size')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')

    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')
    return parser.parse_args()

mb2_setups = [112007, 112009,112011,112014,112024,112026,112028,112030,112032,112033,112034,112035,112036,112037,112038,112039,112040,112041,112042,112043,112047,112048,112049,112050,112051,112052,112053,112055,112057,112058,112059,112061,112067,112069,112071,112073]

if __name__ == '__main__':
    print('reach here A1')
    args = get_args()
    if args.gpu is not None:
        gpu_id = int(args.gpu)
    else:
        gpu_id = 1
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUS", gpus)
    print('reach here A2')

    if gpus and len(gpus)>1:
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        ngpus = len(gpus)
    else:
        ngpus = 1
    train_path = args.save_path
    if not os.path.isdir(train_path):
        os.makedirs(train_path)

    with open(os.path.join(train_path, 'command.txt'), 'w') as f:
        f.write(sys.argv[0])
        f.write(str(sys.argv))

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 1000)

    res_dict = {}
    cms = {}
    res = {}
    all_gt = []
    all_pred = []

    setups = []

    all_data = {}
    training_data = {}
    val_data = {}

    data_files = {}
    label_files = {}
    valid_files = {}
    # if args.augment_with_chest:
    #     folders = {'nwh': args.load_path_nw, 'mb': args.load_path_mb, 'mb_chest': args.load_path_ch}
    # else:
    folders = {'nwh': args.load_path_nw, 'mb': args.load_path_mb}
    data_type = ['X', 'y', 'valid']
    for k, v in folders.items():
        data_files[k] = fnmatch.filter(os.listdir(v), '*_X.*')
        label_files[k] = fnmatch.filter(os.listdir(v), '*_y.*')
        #valid_files[k] = fnmatch.filter(os.listdir(v), '*_valid.*')



    for k, v in folders.items():
        for j, d in enumerate([data_files, label_files]):
            for i, fn in enumerate(d[k]):

                sess = int(fn[0:fn.find('_')])
                setups.append(sess)
                if int(sess) not in all_data.keys():
                    all_data[sess] = {}
                # if 'chest' in k and data_type[j] == 'X':
                #     all_data[sess]['X_chest'] = np.load(os.path.join(folders[k], fn), allow_pickle=True)
                # if 'chest' in k and data_type[j] == 'y':
                #     all_data[sess]['y_chest'] = np.load(os.path.join(folders[k], fn), allow_pickle=True)
                # else:
                all_data[sess][data_type[j]] = np.load(os.path.join(folders[k], fn), allow_pickle=True)
                if np.isnan(all_data[sess][data_type[j]]).any():
                    print(sess, data_type[j], "contains nan")

    usetups = np.unique(setups)

    completed_setups = []
    ngpus=3
    print("ngpus", ngpus)
    if ngpus:
        print('here len(usetups)')
        print(len(usetups))
        rounded_n = int(len(usetups) / ngpus) * ngpus
        print ('here int(len(usetups) / ngpus) * ngpus', int(len(usetups) / ngpus) * ngpus)
        rsetups = np.stack(usetups[:rounded_n])
        print('here         rsetups = np.stack(usetups[:rounded_n])', np.stack(usetups[:rounded_n]))
        print(len(rsetups))
        rsetups = rsetups.reshape(ngpus, int(len(rsetups) / ngpus))



    setups_to_run_on = setups if ngpus == 0 else rsetups[gpu_id, :]
    setups_to_run_on = mb2_setups
    print("warning: running only on mb2 setups")
    print(setups_to_run_on)
    # for k_fold_test_setup in [6284]:
    # if k_fold_test_setup in completed_setups or k_fold_test_setup not in mb2_setups:
    #     print(k_fold_test_setup, "already processed")
    #     continue
    # k_fold_test_setup = 6284
    training_data = {}
    val_data = {}
    print('reach here')
    # print("::::::::::::::::::::", k_fold_test_setup, "::::::::::::::::::::")
    k_fold_test_setup = 'nw+mb1'
    json_fn = '/model/' + str(k_fold_test_setup) + '_model.json'
    weights_fn = '/model/' + str(k_fold_test_setup) + '_model.hdf5'
    print(k_fold_test_setup)
    # db.update_mysql_db(k_fold_test_setup)

    # print(k_fold_test_setup, db.mysql_db)
    # main_session = db.session_from_setup(k_fold_test_setup)
    # radar_setups_all = db.setups_by_session(main_session)
    train_sessions = [s for s in all_data.keys()]
    val_sessions = train_sessions
    excluded_sessions = []#radar_setups_all
    test_sessions = mb2_setups#[k_fold_test_setup]
    #print("all_data.keys()", all_data.keys())
    print("train", train_sessions)
    print('reach here 2')

    # test_sessions = excluded_sessions

    print("test", test_sessions)
    print("excluded", excluded_sessions)

    for tr in train_sessions:
        training_data[tr] = {}
        training_data[tr]['X'] = all_data[tr]['X']#[all_data[tr]['valid'] == 1]
        training_data[tr]['y'] = all_data[tr]['y']#[all_data[tr]['valid'] == 1]

    epochs = 100
    batch_size = args.bs
    samples_per_epoch = 320
    sig = all_data[tr]['X'][0]
    input_sig_shape = sig.shape
    if len(sig.shape) == 1:
        input_sig_shape = (sig.shape[0], 1)
    print('reach here 3')


    model = large_resnet_model_late_batchnorm_1D(input_sig_shape)

    print('reach here 4')
    sname = str(k_fold_test_setup)
    if not os.path.isdir(train_path + '/checkpoints/'):
        os.makedirs(train_path + '/checkpoints/')
    print('reach here 5')

    model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_lr=0.0001, verbose=1)
    print('reach here 6')

    checkpointer = ModelCheckpoint(
        filepath=train_path + '/checkpoints/' + sname + '_model.hdf5', verbose=1,
        save_best_only=True)
    log_dir = train_path + "/" + sname + "_logs/fit/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print('reach here 7')

    model_json = model.to_json()
    with open(train_path + '/checkpoints/' + sname + 'model.json', "w") as json_file:
        json_file.write(model_json)
    failed_load = False

    if args.reload:
        try:
            json_file = open(train_path + json_fn, 'r')
            print("reloading model from", train_path + json_fn)
            model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(model_json)
            # load weights into new model
            print("reloading weights from", train_path + weights_fn)
            model.load_weights(train_path + weights_fn)
            model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
            print(sname, "successfully loaded model from storage")
        except:
            print(sname, "model not found")
            failed_load = True
    plot_model(model, to_file=os.path.join(args.save_path, 'model_plot.png'), show_shapes=True,
               show_layer_names=True)
    print("plot model saved")
    if not args.reload or failed_load:
        try:
            model = large_resnet_model_late_batchnorm_1D(input_sig_shape)
            model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
            print('before model fit')
            model.fit(my_balanced_augmenting_generator(training_data, batch_size),
                  validation_data=my_balanced_augmenting_generator(training_data, batch_size),
                  steps_per_epoch=samples_per_epoch,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[checkpointer, EarlyStopping(patience=args.patience), reduce_lr],
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
        except:
            print("problem with fit")

    #for setup_of_subject in radar_setups_all:  # [110606, 110607, 110608, 110609, 110610, 110611]:#radar_setups_all:
    for setup_of_subject in mb2_setups:  # [110606, 110607, 110608, 110609, 110610, 110611]:#radar_setups_all:
        # mb_setups = MB_HQ
        # nw_setups = NW_HQ

        # if setup_of_subject not in mb2_setups and setup_of_subject not in nw_setups and setup_of_subject not in mb_setups :
        #     continue

        print(setup_of_subject, " TEST using loaded data----------------------------------------")

        time_chunk = model.input_shape[1]
        if setup_of_subject in all_data.keys():
            data_test = all_data[setup_of_subject]['X']
            labels_test = all_data[setup_of_subject]['y']
            #valid = all_data[setup_of_subject]['valid']

            preds = model.predict(data_test)
            preds = preds.reshape([len(preds)])

            print(labels_test)
            print(np.round(preds))
            #print(valid)
            print("gt", 4*np.mean(labels_test), "preds", 4*np.mean(preds))
            res_dict[setup_of_subject] = [4*np.mean(labels_test), 4*np.mean(preds)]
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_gt.npy'), labels_test, allow_pickle=True)
            #np.save(os.path.join(args.save_path, str(setup_of_subject) + '_valid.npy'), valid, allow_pickle=True)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_pred.npy'), preds, allow_pickle=True)
            completed_setups.append(setup_of_subject)
            print(res_dict)
    model = None