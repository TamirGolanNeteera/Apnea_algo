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

from Tests.NN.nn_models import create_vit_VS_regressor_1d


def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')



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
            y = np.load(os.path.join(data_dict['load_path'], filenames[1]), allow_pickle=True)

            if len(y[y<35]) > 0:
                continue

            j = np.random.choice(len(y), stp, replace=False)#choose 8 entries
            if X.shape[1] != 2500:
                continue
            x_out.append(X[j,:])
            y_out.append(y[j])
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
    mean_dict = {}

    if not (os.path.isfile(os.path.join(args.load_path, "file_dict.pkl")) and os.path.isfile(os.path.join(args.load_path, "ys.npy"))):
        for i, fn in enumerate(data_files):
            if i%100 == 0:
                #continue
                print(i,"/",len(data_files))
            sess = int(fn[0:fn.find('_')])
            if sess not in file_dict.keys():
                file_dict[sess] = {}

            y_fn = [f for f in label_files if fn[0:fn.rfind('_')] in f][0]
            s_fn = [f for f in stat_files if fn[0:fn.rfind('_')] in f][0]
            h_fn = [f for f in hq_files if fn[0:fn.rfind('_')] in f][0]
            print( [fn, y_fn, s_fn, h_fn])
            file_dict[sess] = [fn, y_fn, s_fn, h_fn]

            fn_mean = fn[0:fn.find('_')]+'_mean_y.npy'
            y_mean = np.load(os.path.join(args.load_path, fn_mean))
            #print([fn, y_fn], y_mean)
            mean_dict[sess] = y_mean
        ys = np.stack([y for y in mean_dict.values()])
        np.save(os.path.join(args.load_path, "ys.npy"), ys, allow_pickle=True)
        with open(os.path.join(args.load_path, "file_dict.pkl"), 'wb') as fp:
            pickle.dump(file_dict, fp)
        with open(os.path.join(args.load_path, "mean_dict.pkl"), 'wb') as fp:
            pickle.dump(mean_dict, fp)
        print('dictionary saved successfully to file')
    else:
        ys = np.load(os.path.join(args.load_path, "ys.npy"), allow_pickle=True)
        with open(os.path.join(args.load_path, "file_dict.pkl"), 'rb') as fp:
                file_dict = pickle.load(fp)
        with open(os.path.join(args.load_path, "mean_dict.pkl"), 'rb') as fp:
            mean_dict = pickle.load(fp)

    eps = 1e-9
            #ys = np.stack([y for y in mean_dict.values()])

    #np.save(os.path.join(args.save_path, "ys.npy", allow_pickle=True))
    hr_values = [0, 60, 73, 85, 300]
    hist_counts, hist_bins = np.histogram(ys, bins=hr_values)
    weights = hist_counts / np.sum(hist_counts)
    inv_weights = np.array([1.0 / (w + eps) for w in weights])

    weights = inv_weights / np.sum(inv_weights)
    print(weights)

    datasets = [ 'szmc_clinical_trials','mild_motion','cen_exel','N130P_clinical_validatin' ]

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
                if mean_dict[t] < h:
                    setups_by_hr[h].append(t)
                    break

        for t in test_sessions:
            for h in hr_values[1:]:
                try:
                    if mean_dict[t] < h:
                        setups_by_hr_test[h].append(t)
                        break
                except:
                    print(t, "***")
                    continue

        print(setups_by_hr)

        epochs = 500
        batch_size = 256
        samples_per_epoch = 320
        #sig = all_data[tr]['X'][0].T

        input_shape = (2500, 1)
        model = create_vit_VS_regressor_1d(input_shape)
        #model.summary()

        if not os.path.isdir(train_path + '/checkpoints/'):
            os.makedirs(train_path + '/checkpoints/')
        model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=30, min_lr=0.0001, verbose=1)
        checkpointer = ModelCheckpoint(
            filepath=train_path + '/checkpoints/' + ds + '_model.hdf5', verbose=1,
            save_best_only=True)
        log_dir = train_path + "/" + ds + "_logs/fit/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
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
                model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
                print(ds, "successfully loaded model from storage")
            except:
                print(ds, "model not found")
                failed_load = True


        for k,v in setups_by_hr.items():
            print(k, len(v))
        data_dict = {'weights':weights, 'setup_list': setups_by_hr, 'data':train_dict, 'load_path':args.load_path}
        data_dict_test = {'weights':weights, 'setup_list': setups_by_hr_test, 'data':test_dict, 'load_path':args.load_path}

        if not args.reload or failed_load:
            try:
                model.compile(loss=args.loss, optimizer=keras.optimizers.Adam())
                model.fit(my_weighted_generator(data_dict, batch_size),
                      validation_data=my_weighted_generator(data_dict, 64),
                      steps_per_epoch=samples_per_epoch,
                      epochs=epochs,
                      verbose=1,
                      class_weight=None,
                      workers=1,
                      shuffle=True,
                      validation_steps=10,
                      callbacks=[checkpointer, EarlyStopping(patience=args.patience), reduce_lr])
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

            if args.plot:
                fig, ax = plt.subplots(nrows=1, ncols=3,figsize=[30,10])
            for i,setup_of_ds in enumerate(test_sessions):
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
                except:
                    print(setup_of_ds, "failed")

                y_true = np.load(os.path.join(data_dict['load_path'], y_fn), allow_pickle=True)
                stat = np.array(np.load(os.path.join(data_dict['load_path'], s_fn), allow_pickle=True))
                hq = np.array(np.load(os.path.join(data_dict['load_path'], h_fn), allow_pickle=True))

                if X.shape[1] != 2500:
                    continue
                try:
                    y_pred = model.predict(np.stack(X), verbose=0).flatten()
                except:
                    print("no X")
                    continue
                stat[stat == 'normal'] = 'full'
                stat[stat == 'running'] = 'full'

                gt_vec = np.stack(y_true).flatten()
                pred_vec = np.stack(y_pred).flatten()



                diff = np.abs(pred_vec-np.mean(pred_vec))
                # print(np.round(pred_vec),2)
                # print(np.round(diff),2)
                # print(hq)
                hq[diff > 15] = 0
                err = np.abs(np.nanmean(gt_vec) - np.mean(pred_vec))
                serr = np.abs(np.nanmean(gt_vec[stat == 'full']) - np.mean(pred_vec[stat == 'full']))
                herr = np.abs(np.nanmean(gt_vec[hq == 1]) - np.mean(pred_vec[hq == 1]))
                errs.append(err)
                serrs.append(serr)
                herrs.append(herr)
                if dist < 250:
                    back_err.append(err)
                elif dist >= 500:
                    front_err.append(err)
                print(i,"/",len(test_sessions),setup_of_ds, dist,"mm gt", np.mean(np.stack(y_true).flatten()), "pred",np.mean(np.stack(y_pred).flatten()))
                res_fn_y_true = ds+"_"+str(setup_of_ds)+'_y_true.npy'
                res_fn_y_pred = ds+"_"+str(setup_of_ds)+'_y_pred.npy'
                if len(gt_vec[gt_vec < 35])>0:
                    continue
                if args.plot:
                    # ax[0].scatter(np.mean(gt_vec), np.mean(pred_vec), s=5, c=col[i % len(col)])
                    # ax[1].scatter(np.mean(gt_vec[hq==1]), np.mean(pred_vec[hq==1]), s=7, c=col[i % len(col)])
                    # ax[2].scatter(np.mean(gt_vec[stat=='full']), np.mean(pred_vec[stat=='full']), s=5, c=col[i % len(col)])
                    ax[0].scatter(gt_vec, pred_vec, s=2, c=col[i % len(col)])
                    ax[1].scatter(gt_vec[hq == 1], pred_vec[hq == 1], s=2, c=col[i % len(col)])
                    ax[2].scatter(gt_vec[stat == 'full'], pred_vec[stat == 'full'], s=2, c=col[i % len(col)])

                np.save(os.path.join(args.save_path,res_fn_y_true), y_true)
                np.save(os.path.join(args.save_path,res_fn_y_pred), y_pred)

            front_err_mean = np.round(np.mean(front_err), 2)
            back_err_mean = np.round(np.mean(back_err), 2)
            err_mean = np.round(np.nanmean(np.stack(errs)), 2)
            stat_err_mean = np.round(np.nanmean(np.stack(serrs)), 2)
            hqerr_mean = np.round(np.nanmean(np.stack(herrs)), 2)
            print(err_mean, stat_err_mean, hqerr_mean)
            if args.plot:
                ax[0].axline((40, 40), slope=1, linewidth=0.5, c='gray')
                ax[0].axline((45, 40), slope=1, linewidth=0.5, c='gray')
                ax[0].axline((40, 45), slope=1, linewidth=0.5, c='gray')
                ax[1].axline((40, 40), slope=1, linewidth=0.5, c='gray')
                ax[1].axline((45, 40), slope=1, linewidth=0.5, c='gray')
                ax[1].axline((40, 45), slope=1, linewidth=0.5, c='gray')
                ax[2].axline((40, 40), slope=1, linewidth=0.5, c='gray')
                ax[2].axline((45, 40), slope=1, linewidth=0.5, c='gray')
                ax[2].axline((40, 45), slope=1, linewidth=0.5, c='gray')
                ax[0].set_title("Fast "+ds + ' err(all) : '+str(err_mean))
                ax[1].set_title("Fast "+ds + ' err(hq==1): '+str(hqerr_mean))
                ax[2].set_title("Fast "+ds + ' err(stat==full): '+str(stat_err_mean))
                for ii in range(3):
                    ax[ii].set_xlabel('gt')
                    ax[ii].set_ylabel('pred')
                plt.savefig(os.path.join(args.save_path,"out_" + ds + "_point_deviation.png"))
                print("saved image")
                plt.close()
