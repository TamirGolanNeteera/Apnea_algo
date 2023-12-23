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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv2D, Conv1D, GRU, LSTM,  Bidirectional, MaxPool1D, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
import os

import fnmatch
from sklearn import preprocessing

import matplotlib.colors as mcolors

from Tests.vsms_db_api import DB
db = DB()
col = list(mcolors.cnames.keys())
import numpy as np
import scipy as sci
import scipy.signal as sig
import pandas as pd

import time as time
from Tests.NN.nn_models import small_vgg_model, large_vgg_model_late_batchnorm

def rollavg_direct(a,n):
    'Direct "for" loop'
    assert n%2==1
    b = a*0.0
    for i in range(len(a)) :
        b[i]=a[max(i-n//2,0):min(i+n//2+1,len(a))].mean()
    return b

def rollavg_comprehension(a,n):
    'List comprehension'
    assert n%2==1
    r,N = int(n/2),len(a)
    return np.array([a[max(i-r,0):min(i+r+1,N)].mean() for i in range(N)])

def rollavg_convolve(a,n):
    'scipy.convolve'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float')/n, 'same')[n//2:-n//2+1]

def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')

def rollavg_cumsum(a,n):
    'numpy.cumsum'
    assert n%2==1
    cumsum_vec = np.cumsum(np.insert(a, 0, 0))
    return (cumsum_vec[n:] - cumsum_vec[:-n]) / n

def rollavg_cumsum_edges(a,n):
    'numpy.cumsum, edge handling'
    assert n%2==1
    N = len(a)
    cumsum_vec = np.cumsum(np.insert(np.pad(a,(n-1,n-1),'constant'), 0, 0))
    d = np.hstack((np.arange(n//2+1,n),np.ones(N-n)*n,np.arange(n,n//2,-1)))
    return (cumsum_vec[n+n//2:-n//2+1] - cumsum_vec[n//2:-n-n//2]) / d

def rollavg_roll(a,n):
    'Numpy array rolling'
    assert n%2==1
    N = len(a)
    rolling_idx = np.mod((N-1)*np.arange(n)[:,None] + np.arange(N), N)
    return a[rolling_idx].mean(axis=0)[n-1:]

def rollavg_roll_edges(a,n):
    # see https://stackoverflow.com/questions/42101082/fast-numpy-roll
    'Numpy array rolling, edge handling'
    assert n%2==1
    a = np.pad(a,(0,n-1-n//2), 'constant')*np.ones(n)[:,None]
    m = a.shape[1]
    idx = np.mod((m-1)*np.arange(n)[:,None] + np.arange(m), m) # Rolling index
    out = a[np.arange(-n//2,n//2)[:,None], idx]
    d = np.hstack((np.arange(1,n),np.ones(m-2*n+1+n//2)*n,np.arange(n,n//2,-1)))
    return (out.sum(axis=0)/d)[n//2:]

def rollavg_pandas(a,n):
    'Pandas rolling average'
    return pd.DataFrame(a).rolling(n, center=True, min_periods=1).mean().to_numpy()

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
            y_out.append(v['y'][cur_select][0])

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)

        yield x_out, y_out

probs = {60:1, 70:0, 80:0, 90:0, 100:0, 110:0, 120:0.5, 130:0.5, 140:0, 150:0, 160:0, 170:0, 180:0, 190:0, 200:0, 210:0, 220:0}
def my_weighted_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        sessions = [s for s in data_dict.keys()]

        np.random.shuffle(sessions)

        for k in range(batchsize):
            i = np.random.choice(len(sessions), 1)[0]
            v = data_dict[sessions[i]]
            cur_select = np.random.choice(len(v['y']), 1)[0]

            if v['y'][cur_select][0]>119 and v['y'][cur_select][0] < 133:
                if np.random.random() < 0.5:
                    continue

            x_out.append(v['X'][cur_select])
            y_out.append(v['y'][cur_select][0])

        x_out = np.asarray(x_out)
        y_out = np.asarray(y_out)

        yield x_out, y_out

#TODO:
#make generator that loads setup feature vector, selects random location and crops sample vector


def my_balanced_augmenting_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        sessions = [s for s in data_dict.keys()]

        np.random.shuffle(sessions)

        for k in range(batchsize):
            i = np.random.choice(len(sessions), 1)[0]
            v = data_dict[sessions[i]]
            cur_select = np.random.choice(len(v['y']), 1)[0]
            y = v['y'][cur_select]
            if y > 5 and np.random.rand() > 0.5:
                x_o = np.zeros_like(v['X'][cur_select])

                half = int(len(x_o)/2)

                x_o[:half] = v['X'][cur_select][half:]
                x_o[half:] = v['X'][cur_select][:half]
                x_out.append(x_o)
            else:
                if y == 0 and np.random.rand() > 0.5:
                    continue
                else:
                    x_out.append(v['X'][cur_select])
            y_out.append(v['y'][cur_select])
            #print(i, cur_select, len(x_out), len(y_out))

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
    parser.add_argument('-seed', metavar='seed', type=int, required=False, help='Set seed for random')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('-gpu', metavar='gpu', type=int, required=False,help='gpu device id')

    parser.add_argument('--augment', action='store_true', help='augmentation')
    parser.add_argument('--reverse', action='store_true', help='reverse setup order')
    parser.add_argument('-patience', metavar='window', type=int, required=True, help='when to stop training')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.gpu:
        gpu_id = int(args.gpu)
    else:
        gpu_id = 0
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUS", gpus)
    if gpus:
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')

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

    data_files = fnmatch.filter(os.listdir(args.load_path), '*_X.*')
    label_files = fnmatch.filter(os.listdir(args.load_path), '*_y.*')

    setups = []
    completed_setups = []
    completed_subjects = []
    for f in data_files:
        setups.append(int(f[0:f.find('_')]))

    print(setups)

    if args.reverse:
        setups = setups[::-1]

    for k_fold_test_setup in setups:
        if k_fold_test_setup in completed_setups:
            print(k_fold_test_setup, "already processed")
            continue
        all_data = {}
        training_data = {}
        val_data = {}

        db.update_mysql_db(k_fold_test_setup)

        for i, fn in enumerate(data_files):
            sess = int(fn[0:fn.find('_')])

            if int(sess) not in all_data.keys():
                all_data[sess] = {}
            all_data[sess]['X'] = np.load(os.path.join(args.load_path, fn), allow_pickle=True)
            if np.isnan(all_data[sess]['X']).any():
                print(sess, "X contains nan")

        for i, fn in enumerate(label_files):
            sess = int(fn[0:fn.find('_')])

            if sess not in all_data.keys():
                continue
            all_data[sess]['y'] = np.load(os.path.join(args.load_path, fn), allow_pickle=True)
            if np.isnan(all_data[sess]['y']).any():
                print(sess, "y contains nan")
        sname = db.setup_subject(k_fold_test_setup)
        print("::::::::::::::::::::", k_fold_test_setup, sname,"::::::::::::::::::::")

        json_fn = '/model/' + sname + '_model.json'
        weights_fn = '/model/' + sname + '_model.hdf5'

        all_subject_setups = [s for s in setups if db.setup_subject(s) == sname]
        train_sessions = [s for s in setups if s not in all_subject_setups]# and s in all_setups]
        excluded_sessions = all_subject_setups

        test_sessions = [k_fold_test_setup]
        # print("all_data.keys()",all_data.keys())
        # print("train", train_sessions)
        print("test", test_sessions)
        print("excluded", excluded_sessions)
        for tr in train_sessions:
            training_data[tr] = {}
            training_data[tr]['X'] = all_data[tr]['X'].reshape(all_data[tr]['X'].shape[0], all_data[tr]['X'].shape[2], all_data[tr]['X'].shape[1])#[all_data[tr]['valid'] == 1]
            training_data[tr]['y'] = all_data[tr]['y']#[all_data[tr]['valid'] == 1]
        #print("training_data.keys()",training_data.keys())
        sname = sname.replace(' ', '_')
        epochs = 500
        batch_size = 128
        samples_per_epoch = 320
        #sig = all_data[tr]['X'][0].T

        input_shape = (5000, 8,1)

        model = large_vgg_model_late_batchnorm(input_shape)


        if not os.path.isdir(train_path + '/checkpoints/'):
            os.makedirs(train_path + '/checkpoints/')
        model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())

        checkpointer = ModelCheckpoint(
            filepath=train_path + '/checkpoints/' + sname + '_model.hdf5', verbose=1,
            save_best_only=True)
        log_dir = train_path + "/" + str(k_fold_test_setup) + "_logs/fit/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_json = model.to_json()
        with open(train_path + '/checkpoints/' + sname + 'model.json', "w") as json_file:
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
            model.fit(my_balanced_generator(training_data, batch_size),

                  #validation_data=my_generator(data_train, labels_train, batch_size),
                  validation_data=my_balanced_generator(training_data, 128),
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


        # test the NN

        for setup_of_subject in all_subject_setups:
            completed_setups.append(setup_of_subject)

            print(setup_of_subject," TEST ", db.setup_subject(setup_of_subject), "----------------------------------------")

            data_test = all_data[setup_of_subject]['X'].reshape(all_data[setup_of_subject]['X'].shape[0], all_data[setup_of_subject]['X'].shape[2], all_data[setup_of_subject]['X'].shape[1])
            preds = model.predict(data_test)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_gt.npy'), all_data[setup_of_subject]['y'], allow_pickle=True)
            #np.save(os.path.join(args.save_path, str(setup_of_subject) + '_valid.npy'), valid, allow_pickle=True)
            np.save(os.path.join(args.save_path, str(setup_of_subject) + '_pred.npy'), preds, allow_pickle=True)
            preds =preds.flatten()
            print(preds)
            print(all_data[setup_of_subject]['y'][:,0])
            print(np.mean(preds), np.mean(all_data[setup_of_subject]['y'][:,0]))
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(all_data[setup_of_subject]['y'][:,0], label='ref')
            plt.plot(preds, label='pred')
            preds_avg = rollavg_convolve_edges(preds, 9)
            #preds_avg2 = rollavg_convolve_edges(preds, 29)
            plt.plot(preds_avg, label='pred_avg')
            #plt.plot(preds_avg2, label='pred_avg_smooth')
            plt.legend()
            error = np.mean(np.abs(preds-all_data[setup_of_subject]['y'][:,0]))
            plt.title(str(setup_of_subject)+' '+db.setup_subject(setup_of_subject)+ ' Systolic, error ' + str(np.round(error,2)))
            plt.savefig(os.path.join(args.save_path, str(setup_of_subject)+'_'+db.setup_subject(setup_of_subject)+'_S_bn.png'))

            plt.close()
            res[setup_of_subject] = error
        completed_subjects.append(db.setup_subject(setup_of_subject))
        print(res)