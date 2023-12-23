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
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import fnmatch
import pickle
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
col = list(mcolors.cnames.keys())
from Tests.vsms_db_api import DB
db = DB()
col = list(mcolors.cnames.keys())
import numpy as np
import scipy as sci

from Tests.NN.nn_models import create_vit_classifier_1d


def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')

stat_dict = {'normal':1, 'empty':0, 'zrr':1, 'motion':1}

import random


def my_weighted_generator(data_dict, batchsize):

    while 1:
        x_out = []
        y_out = []
        stp=8
        category = random.choices(list(data_dict['setup_list'].keys()), weights=data_dict['weights'], k=batchsize)
        k=0
        while k < batchsize:
            sessions = data_dict['setup_list'][category[k]]
            i = np.random.choice(len(sessions), 1)[0]#choose a setup randomly
            filenames = list(data_dict['data'][sessions[i]])
            X = np.load(os.path.join(data_dict['load_path'], filenames[0]), allow_pickle=True)
            s = np.load(os.path.join(data_dict['load_path'], filenames[2]), allow_pickle=True)

            j = np.random.choice(len(s), stp, replace=False)#choose 8 entries
            if X.shape[1] != 2500:
                continue
            sj_int = np.zeros(stp)
            for st in range(stp):
                sj_int[st] = stat_dict[s[j][st]]
            x_out.append(X[j,:])
            y_out.append(sj_int)
            k +=stp

        x_out = np.vstack(x_out)
        y_out = np.hstack(y_out)
        yield x_out, y_out

def my_generator(data_dict, batchsize):

    while 1:
        x_out = []
        y_out = []
        stp=8

        k=0
        while k < batchsize:
            sessions = list(data_dict['data'].keys())

            i = np.random.choice(len(sessions), 1)[0]#choose a setup randomly

            filenames = list(data_dict['data'][sessions[i]])

            X = np.load(os.path.join(data_dict['load_path'], filenames[0]), allow_pickle=True)
            s = np.load(os.path.join(data_dict['load_path'], filenames[2]), allow_pickle=True)

            j = np.random.choice(len(s), stp, replace=False)#choose 8 entries
            if X.shape[1] != 2500:
                continue

            sj_int = np.zeros(stp)
            for st in range(stp):
                sj_int[st] = stat_dict[s[j][st]]

            x_out.append(X[j,:])

            y_out.append(sj_int)
            k +=stp

        x_out = np.vstack(x_out)
        y_out = np.hstack(y_out)
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

    parser.add_argument('--test', action='store_true', help='test NN')
    parser.add_argument('--plot', action='store_true', help='create images of output')
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
    stat_files = fnmatch.filter(os.listdir(args.load_path), '*_stat.*')
    hq_files = fnmatch.filter(os.listdir(args.load_path), '*_hq.*')
    label_files = [l for l in label_files if 'mean' not in l]
    print(len(data_files),len(label_files), len(stat_files), len(hq_files))

    setups = []
    completed_setups = []
    completed_subjects = []
    for f in data_files:
        setups.append(int(f[0:f.find('_')]))

    file_dict = {}
    all_data = {}
    perc_dict = {}

    if True:#not (os.path.isfile(os.path.join(args.load_path, "file_dict.pkl")) and os.path.isfile(os.path.join(args.load_path, "ys.npy"))):
        for i, fn in enumerate(data_files):
            if i%100 == 0:
                #continue
                print(i,"/",len(data_files))
            sess = int(fn[0:fn.find('_')])
            if sess not in file_dict.keys():
                file_dict[sess] = {}

            y_fn = ''#[f for f in label_files if fn[0:fn.rfind('_')] in f][0]
            s_fn = [f for f in stat_files if fn[0:fn.rfind('_')] in f][0]
            h_fn = ''#[f for f in hq_files if fn[0:fn.rfind('_')] in f][0]
            print( [fn, y_fn, s_fn, h_fn])
            file_dict[sess] = [fn, y_fn, s_fn, h_fn]

            fn_perc = fn[0:fn.find('_')]+'_perc_empty.npy'
            empty_perc = np.load(os.path.join(args.load_path, fn_perc))
            #print([fn, y_fn], y_mean)
            perc_dict[sess] = float(empty_perc)
        ys = np.stack([y for y in perc_dict.values()])
        np.save(os.path.join(args.load_path, "ys.npy"), ys, allow_pickle=True)
        with open(os.path.join(args.load_path, "file_dict.pkl"), 'wb') as fp:
            pickle.dump(file_dict, fp)
        with open(os.path.join(args.load_path, "perc_dict.pkl"), 'wb') as fp:
            pickle.dump(perc_dict, fp)
        print('dictionary saved successfully to file')
    else:
        ys = np.load(os.path.join(args.load_path, "ys.npy"), allow_pickle=True)
        with open(os.path.join(args.load_path, "file_dict.pkl"), 'rb') as fp:
            file_dict = pickle.load(fp)
        with open(os.path.join(args.load_path, "perc_dict.pkl"), 'rb') as fp:
            perc_dict = pickle.load(fp)

    eps = 1e-9
            #ys = np.stack([y for y in mean_dict.values()])

    #np.save(os.path.join(args.save_path, "ys.npy", allow_pickle=True))
    hr_values = [0, 0.01, 0.3, 1]
    hist_counts, hist_bins = np.histogram(ys, bins=hr_values)
    weights = hist_counts / np.sum(hist_counts)
    inv_weights = np.array([1.0 / (w + eps) for w in weights])

    weights = inv_weights / np.sum(inv_weights)
    print(weights)

    datasets = ['es_office_benchmark',  'ec_office_benchmark_130P', 'N130P_ec_benchmark', 'ec_benchmark', 'szmc_clinical_trials','mild_motion','cen_exel','N130P_clinical_validatin' ]

    for ds in datasets:
        test_sessions = db.benchmark_setups(ds)
        train_sessions = list(set(setups) - set(test_sessions))
        print(ds, "train setups:", len(train_sessions), "test setups:", len(test_sessions))
        training_data = {}
        val_data = {}

        #db.update_mysql_db(k_fold_test_setup)
        #sname = db.setup_subject(k_fold_test_setup)
        print("::::::::::::::::::::", ds, "::::::::::::::::::::")

        json_fn = '/model/' + ds + '_model.json'
        weights_fn = '/model/' + ds + '_model.hdf5'

        print("test", test_sessions)

        train_dict = {key: file_dict[key] for key in train_sessions if key in file_dict}
        test_dict = {key: file_dict[key] for key in test_sessions if key in file_dict}

        print(len(train_sessions))
        ccc = 0
        all_ys = []

        setups_by_hr = {}
        setups_by_hr_test = {}

        for h in hr_values[1:]:
            setups_by_hr[h] = []
            setups_by_hr_test[h] = []

        for t in train_sessions:
            for h in hr_values[1:]:
                if perc_dict[t] < h:
                    setups_by_hr[h].append(t)
                    break

        for t in test_sessions:
            for h in hr_values[1:]:
                try:
                    if perc_dict[t] < h:
                        setups_by_hr_test[h].append(t)
                        break
                except:
                    print(t, "***")
                    continue

        print(setups_by_hr)

        epochs = 100
        batch_size = 1024
        samples_per_epoch = 320
        #sig = all_data[tr]['X'][0].T

        input_shape = (2500, 1)
        model = create_vit_classifier_1d(input_shape, 2)
        #model.summary()

        if not os.path.isdir(train_path + '/checkpoints/'):
            os.makedirs(train_path + '/checkpoints/')
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics='accuracy')
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=0.0001, verbose=1)
        checkpointer = ModelCheckpoint(monitor='val_accuracy',
            filepath=train_path + '/checkpoints/' + ds + '_model.hdf5', verbose=1,
            save_best_only=True)
        log_dir = train_path + "/" + ds + "_logs/fit/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_json = model.to_json()
        with open(train_path + '/checkpoints/' + ds + 'model.json', "w") as json_file:
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
                model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics='accuracy')
                print(ds, "successfully loaded model from storage")
            except:
                print(ds, "model not found")
                failed_load = True



        data_dict = {'data':train_dict, 'load_path':args.load_path}
        data_dict_test = {'data':test_dict, 'load_path':args.load_path}

        data_dict = {'weights': weights, 'setup_list': setups_by_hr, 'data': train_dict, 'load_path': args.load_path}
        data_dict_test = {'weights': weights, 'setup_list': setups_by_hr_test, 'data': test_dict, 'load_path': args.load_path}

        if not args.reload or failed_load:
            try:
                model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics='accuracy')
                model.fit(my_weighted_generator(data_dict, batch_size),
                      validation_data=my_weighted_generator(data_dict_test, 128),
                      steps_per_epoch=samples_per_epoch,
                      epochs=epochs,
                      verbose=1,
                      class_weight=None,
                      workers=1,
                      shuffle=True,
                      validation_steps=10,
                      callbacks=[checkpointer, EarlyStopping(patience=args.patience)])
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
            front_err = []
            back_err = []
            errs = []
            serrs = []
            herrs = []

            for i,setup_of_ds in enumerate(test_sessions):
                d = db.setup_distance(setup_of_ds)
                if args.plot:
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[30, 10])
                if setup_of_ds == 11255:
                    continue
                if setup_of_ds in train_sessions:
                    print("test in train!!")
                    continue
                all_X = []
                data_files = fnmatch.filter(os.listdir(args.load_path), str(setup_of_ds)+'*_X.*')
                label_files = fnmatch.filter(os.listdir(args.load_path), str(setup_of_ds)+'*_y.*')
                stat_files = fnmatch.filter(os.listdir(args.load_path), str(setup_of_ds)+'*_stat.*')
                hq_files = fnmatch.filter(os.listdir(args.load_path), str(setup_of_ds)+'*_hq.*')

                dist = db.setup_distance(setup_of_ds)
                try:
                    x_fn = data_files[0]
                    X = np.load(os.path.join(args.load_path,x_fn), allow_pickle=True)
                    st = x_fn[:x_fn.find('X')]+'y'
                    y_fn = [g for g in label_files if st in g and 'mean' not in g][0]
                    s_fn = [f for f in stat_files if x_fn[0:x_fn.rfind('_')] in f][0]
                    h_fn = [f for f in hq_files if x_fn[0:x_fn.rfind('_')] in f][0]
                    y_true = np.load(os.path.join(data_dict['load_path'], y_fn), allow_pickle=True)
                    stat = np.array(np.load(os.path.join(data_dict['load_path'], s_fn), allow_pickle=True))
                    hq = np.array(np.load(os.path.join(data_dict['load_path'], h_fn), allow_pickle=True))
                    stat_class = np.array([stat_dict[si] for si in stat])


                except:
                    print(setup_of_ds, "failed")
                    continue

                if X.shape[1] != 2500:
                    continue
                try:
                    y_pred = model.predict(np.stack(X), verbose=0)
                    yp = np.argmax(y_pred, axis=1)
                    acc = 1.0-np.sum(np.abs(stat_class-yp))/len(yp)
                    print(i,"/",len(test_sessions),setup_of_ds," accuracy", acc)
                    #print(yp)
                except:
                    print("no X")
                    continue
                stat[stat == 'normal'] = 'full'
                stat[stat == 'running'] = 'full'

                gt_vec = np.stack(y_true).flatten()
                pred_vec = np.stack(y_pred).flatten()
                diff = np.abs(pred_vec-np.mean(pred_vec))

                print(i,"/",len(test_sessions),setup_of_ds)
                res_fn_y_true = ds+"_"+str(setup_of_ds)+'_y_true.npy'
                res_fn_y_pred = ds+"_"+str(setup_of_ds)+'_y_pred.npy'

                if args.plot:
                    fig.suptitle(ds+" "+str(setup_of_ds)+" dist: "+str(d)+" acc: "+str(acc))
                    ax.plot(stat_class, label='gt', linewidth=2.5, alpha=0.5, c='blue')
                    ax.plot(y_pred[:,1], label='pred', linewidth=0.5, alpha=0.5, c='red')
                    ax.plot(yp, label='pred', linewidth=0.75, alpha=0.5, c='red')
                    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.1)
                    ax.axhline(y=1, color='gray', linewidth=0.5, alpha=0.1)
                    ax.legend()
                    plt.savefig(os.path.join(args.save_path,"out_" + ds + "_"+str(setup_of_ds)+".png"))
                    print("saved image")
                    plt.close()


                np.save(os.path.join(args.save_path,res_fn_y_true), y_true)
                np.save(os.path.join(args.save_path,res_fn_y_pred), y_pred)


