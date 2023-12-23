import argparse
import datetime
import pandas as pd
from sklearn.preprocessing import scale
import random
from Tests.NN.reiniers_create_data import *
from Tests.vsms_db_api import *
from Tests.Utils.DBUtils import calculate_delay
from Tests.Utils.LoadingAPI import load_nes
from pylibneteera.sp_utils import resample_sig
import fnmatch
import json
from Tests.NN.create_apnea_count_AHI_data import compute_phase

def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()

def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=False)

    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save data', required=True)
    parser.add_argument('-load_paths', nargs="+", required=True)

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    data_files = []
    label_files = []
    stat_files = []
    hr_hq_files = []
    all_stat = []
    for p in args.load_paths:
        acc_path = p[:p.rfind('/')+1]+'accumulated'
        print(acc_path)
        df = fnmatch.filter(os.listdir(acc_path), '*_scg.npy')
        lf = fnmatch.filter(os.listdir(acc_path), '*_ref_hr.*')

        dynamic_path = p[:p.rfind('/')+1]+'dynamic'
        print(dynamic_path)
        sf = fnmatch.filter(os.listdir(acc_path), '*_stat*.*')
        hf = fnmatch.filter(os.listdir(dynamic_path), '*_hr_high_quality_indicator*.*')
        df_full = [os.path.join(acc_path, f) for f in df]
        lf_full = [os.path.join(acc_path, f) for f in lf]
        sf_full = [os.path.join(acc_path, f) for f in sf]
        hf_full = [os.path.join(dynamic_path, f) for f in hf]
        data_files.append(df_full)
        label_files.append(lf_full)
        stat_files.append(sf_full)
        hr_hq_files.append(hf_full)

    data_files = np.hstack(data_files)
    label_files = np.hstack(label_files)
    stat_files = np.hstack(stat_files)
    hr_hq_files = np.hstack(hr_hq_files)
    print(len(data_files))
    print(len(label_files))
    print(len(stat_files))
    print(len(hr_hq_files))

    all_setups = []
    completed_setups = []
    completed_subjects = []
    # for f in data_files:
    #     setups.append(int(f[0:f.find('_')]))

    fdict = {}
    for i, fn in enumerate(data_files):
        short_fn = fn[fn.rfind('/')+1:]
        sess_str = short_fn[0:short_fn.find('_')]
        sess = int(sess_str)
        all_setups.append(sess)

        y = None
        try:
            X = np.load(fn, allow_pickle=True)
        except:
            print(sess, fn, "X loading failed")
            continue
        try:
            fn_y = [f for f in label_files if sess_str in f][0]
            y = np.load(fn_y, allow_pickle=True)
        except:
            print(sess, 'failed to find y labels')


        try:
            fn_stat = [f for f in stat_files if sess_str in f][0]
            stat = np.load(fn_stat, allow_pickle=True)
            print("loaded stat", len(stat))
            print(np.unique(stat))
            perc = len(stat[stat=='empty'])/len(stat)
            print("% empty", perc)
            all_stat.append(np.unique(stat))
        except:
            print(sess, "no status data, assuming all full")
            stat = None
            stat = None
            perc = 0
        try:
            fn_hq = [f for f in hr_hq_files if sess_str in f][0]
            hq_str = np.load(fn_hq, allow_pickle=True)
            hq = np.array([int(j) for j in hq_str])
            print("loaded hq", len(hq))
        except:
            print(sess, "no hq data, assuming all hq")
            hq = None


        print(fn, "loaded successfully", X.shape)
        if y is not None:
            idxs = np.array(list(set(X.keys()).intersection(set(y.keys()))))
            if len(idxs) == 0:
                print(sess,"empty idxs")
                continue
            if y is not None:
                fn_mean_y = str(sess) + '_mean_y.npy'
                np.save(os.path.join(args.save_path, fn_mean_y), np.mean(y[y > 0]))

        else:
            idxs = np.array(list(set(X.keys()).intersection(set(stat.keys()))))
            if len(idxs) == 0:
                print(sess,"empty idxs")
                continue

        fn_perc_empty = str(sess) + '_perc_empty.npy'
        np.save(os.path.join(args.save_path, fn_perc_empty), perc)
        fn_X = str(sess)+'_X.npy'
        fn_y = str(sess)+'_y.npy'
        fn_st = str(sess)+'_stat.npy'
        fn_hq = str(sess)+'_hq.npy'


        print(i, fn_X ,fn_y )

        if stat is not None and len(idxs[idxs >= len(stat)]) > 0:
            print("not ok, stat too short")
            continue
        np.save(os.path.join(args.save_path, fn_X), np.stack(X[idxs].to_numpy()), allow_pickle=True)
        if y is not None:
            np.save(os.path.join(args.save_path, fn_y), y[idxs], allow_pickle=True)

        try:
            if hq is not None:
                np.save(os.path.join(args.save_path, fn_hq), hq[idxs], allow_pickle=True)
            else:
                hq = np.ones(len(y[idxs]))
                np.save(os.path.join(args.save_path, fn_hq), hq, allow_pickle=True)
            if stat is not None:
                np.save(os.path.join(args.save_path, fn_st), stat[idxs], allow_pickle=True)
            else:
                st = np.array(['normal'] * len(y[idxs]))
                np.save(os.path.join(args.save_path, fn_st), st, allow_pickle=True)
        except:
            print("NOT OK", sess)
            continue
        completed_setups.append(sess)
    print("Done")



