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
import matplotlib.pyplot as plt
import fnmatch
import random
import matplotlib.colors as mcolors

from Tests.vsms_db_api import DB
db = DB()
col = list(mcolors.cnames.keys())
import numpy as np
import scipy as sci
import pandas as pd

import time as time
from Tests.NN.nn_models import small_vgg_model, large_resnet_model_late_batchnorm, small_resnet_model_late_batchnorm

def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')

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


def my_weighted_generator(data_dict, batchsize):
    while 1:
        x_out = []
        y_out = []

        category = random.choices(list(data_dict['setup_list'].keys()), weights=data_dict['weights'], k=batchsize)
        for k in range(batchsize):

            sessions = data_dict['setup_list'][category[k]]

            i = np.random.choice(len(sessions), 1)[0]

            v = data_dict['data'][sessions[i]]
            cur_select = np.random.choice(len(v['y']), 1)[0]

            x_out.append(v['X'][cur_select])
            y_out.append(v['y'][cur_select][0])

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
    parser.add_argument('-bs', metavar='seed', type=int, required=False, help='Batch size')
    parser.add_argument('--reload', action='store_true', help='reload stored model (no train)')
    parser.add_argument('--scale', action='store_true', help='scale test vectors to m=0, s=1')
    parser.add_argument('-gpu', metavar='gpu', type=int, required=False,help='gpu device id')

    parser.add_argument('--augment', action='store_true', help='augmentation')
    parser.add_argument('--reverse', action='store_true', help='reverse setup order')
    parser.add_argument('--test', action='store_true', help='test NN')
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
        ngpus = len(gpus)
    else:
        ngpus = 0
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

    print(len(data_files))
    print(len(label_files))

    setups = []
    completed_setups = []
    completed_subjects = []
    for f in data_files:
        setups.append(int(f[0:f.find('_')]))

    if args.reverse:
        setups = setups[::-1]

    subjects = [db.setup_subject(s) for s in setups]
    usubjects = np.unique(subjects)


    if ngpus:
        print(len(usubjects))
        rounded_n = int(len(usubjects)/ngpus)*ngpus
        rsubjects = np.stack(usubjects[:rounded_n])
        print(len(rsubjects))
        rsubjects = rsubjects.reshape(ngpus,int(len(rsubjects)/ngpus))

    all_data = {}
    for i, fn in enumerate(data_files):
        sess = int(fn[0:fn.find('_')])
        if int(sess) not in all_data.keys():
            all_data[sess] = {}
        print("loading setup", sess, "data")
        all_data[sess]['X'] = np.load(os.path.join(args.load_path, fn), allow_pickle=True)

    for i, fn in enumerate(label_files):
        sess = int(fn[0:fn.find('_')])
        if sess not in all_data.keys():
            continue
        print("loading setup", sess, "labels")
        all_data[sess]['y'] = np.load(os.path.join(args.load_path, fn), allow_pickle=True)


    subjects_to_run_on = usubjects if ngpus == 0 else rsubjects[args.gpu, :]
    print(subjects_to_run_on)
    for k_fold_test_subject in subjects_to_run_on:
        if k_fold_test_subject in completed_setups:
            print(k_fold_test_subject, "already processed")
            continue

        training_data = {}
        val_data = {}

        #db.update_mysql_db(k_fold_test_setup)
        #sname = db.setup_subject(k_fold_test_setup)
        print("::::::::::::::::::::", k_fold_test_subject, "::::::::::::::::::::")

        json_fn = '/model/' + k_fold_test_subject + '_model.json'
        weights_fn = '/model/' + k_fold_test_subject + '_model.hdf5'

        excluded_sessions = [s for s in setups if db.setup_subject(s) == k_fold_test_subject]
        train_sessions = [s for s in setups if s not in excluded_sessions]# and s in all_setups]

        setups_by_bp = {90:[], 100: [], 110: [], 120: [], 130: [], 140: [], 150: [], 160: [], 170: []}

        print(len(train_sessions))
        ccc = 0
        for sess in train_sessions:

            mean_y = np.mean(all_data[sess]['y'][:,0])

            if mean_y >=170:
                 setups_by_bp[170].append(sess)
                 #print(sess, mean_y, "to 170")
            else:
                for k in range(90, 170, 10):
                    if mean_y >= k and mean_y < k+10:
                        setups_by_bp[k].append(sess)
                        #(sess, mean_y, "to", k)
            ccc+=1
        print(ccc)
        print(setups_by_bp)

        test_sessions = excluded_sessions
        # print("all_data.keys()",all_data.keys())
        # print("train", train_sessions)
        print("test", test_sessions)
        print("excluded", excluded_sessions)
        for tr in train_sessions:
            training_data[tr] = {}
            training_data[tr]['X'] = all_data[tr]['X'].reshape(all_data[tr]['X'].shape[0], all_data[tr]['X'].shape[2], all_data[tr]['X'].shape[1])#[all_data[tr]['valid'] == 1]
            training_data[tr]['y'] = all_data[tr]['y']#[all_data[tr]['valid'] == 1]
        #print("training_data.keys()",training_data.keys())
        sname = k_fold_test_subject.replace(' ', '_')
        epochs = 100
        batch_size = args.bs
        samples_per_epoch = 320
        #sig = all_data[tr]['X'][0].T

        input_shape = (all_data[tr]['X'][0].shape[1], all_data[tr]['X'][0].shape[0], 1)
        model = large_resnet_model_late_batchnorm(input_shape)
        model.summary()

        if not os.path.isdir(train_path + '/checkpoints/'):
            os.makedirs(train_path + '/checkpoints/')
        model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=10, min_lr=0.0001, verbose=1)
        checkpointer = ModelCheckpoint(
            filepath=train_path + '/checkpoints/' + sname + '_model.hdf5', verbose=1,
            save_best_only=True)
        log_dir = train_path + "/" + sname + "_logs/fit/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
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


        eps=1e-9
        sum_setups = len(np.hstack(setups_by_bp.values()))
        weights = [len(v)/sum_setups for v in list(setups_by_bp.values())]
        print(weights)
        inv_weights = np.array([1.0/(w+eps) for w in weights])

        weights  = inv_weights/np.sum(inv_weights)


        #weights = 0.25-np.array(weights)
        print(weights)

        for k,v in setups_by_bp.items():
            print(k, len(v))
        data_dict = {'weights':weights, 'setup_list': setups_by_bp, 'data':training_data}

        if not args.reload or failed_load:
            try:
                model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
                model.fit(my_weighted_generator(data_dict, batch_size),
                      validation_data=my_weighted_generator(data_dict, 32),
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



    # test the NN
        if args.test:
            for setup_of_subject in excluded_sessions:
                completed_setups.append(setup_of_subject)

                print(setup_of_subject," TEST ", db.setup_subject(setup_of_subject), "----------------------------------------")

                data_test = all_data[setup_of_subject]['X'].reshape(all_data[setup_of_subject]['X'].shape[0], all_data[setup_of_subject]['X'].shape[2], all_data[setup_of_subject]['X'].shape[1])
                preds = model.predict(data_test)
                np.save(os.path.join(args.save_path, str(setup_of_subject) + '_gt.npy'), all_data[setup_of_subject]['y'], allow_pickle=True)
                #np.save(os.path.join(args.save_path, str(setup_of_subject) + '_X.npy'), data_test, allow_pickle=True)
                np.save(os.path.join(args.save_path, str(setup_of_subject) + '_pred.npy'), preds, allow_pickle=True)
                print('saved setup data, labels and preds')
                print(os.path.join(args.save_path, str(setup_of_subject) + '_gt.npy'))
                #print(os.path.join(args.save_path, str(setup_of_subject) + '_X.npy'))
                print(os.path.join(args.save_path, str(setup_of_subject) + '_pred.npy'))
                preds =preds.flatten()
                print(preds)
                print(all_data[setup_of_subject]['y'][:,0])
                print(np.mean(preds), np.mean(all_data[setup_of_subject]['y'][:,0]))

                # plt.figure()
                # plt.plot(all_data[setup_of_subject]['y'][:,0], label='ref')
                # plt.plot(preds, label='pred')
                preds_avg = rollavg_convolve_edges(preds, 9)
                #preds_avg2 = rollavg_convolve_edges(preds, 29)
                # plt.plot(preds_avg, label='pred_avg')
                #plt.plot(preds_avg2, label='pred_avg_smooth')
                # plt.legend()
                error = np.mean(np.abs(preds-all_data[setup_of_subject]['y'][:,0]))
                # plt.title(str(setup_of_subject)+' '+db.setup_subject(setup_of_subject)+ ' Systolic, error ' + str(np.round(error,2)))
                # plt.savefig(os.path.join(args.save_path, str(setup_of_subject)+'_'+db.setup_subject(setup_of_subject)+'_S_bn.png'))
                # #
                # plt.close()
                res[setup_of_subject] = error
        completed_subjects.append(db.setup_subject(setup_of_subject))
        print(res)