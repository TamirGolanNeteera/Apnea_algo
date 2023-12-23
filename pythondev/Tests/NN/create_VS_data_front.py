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
    parser.add_argument('-load_path', metavar='SavePath', type=str, help='Path to save data', required=True)

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    data_files = fnmatch.filter(os.listdir(args.load_path), '*_scg.npy')
    label_files = fnmatch.filter(os.listdir(args.load_path), '*_ref_hr.*')
    dynamic_path = args.load_path[:args.load_path.rfind('/')+1]+'dynamic'
    stat_files = fnmatch.filter(os.listdir(args.load_path), '*_stat*.*')
    hr_hq_files = fnmatch.filter(os.listdir(dynamic_path), '*_hr_high_quality_indicator*.*')
    print(len(data_files))
    print(len(label_files))
    print(len(stat_files))
    print(len(hr_hq_files))

    setups = []
    completed_setups = []
    completed_subjects = []
    for f in data_files:
        setups.append(int(f[0:f.find('_')]))

    fdict = {}
    for i, fn in enumerate(data_files):
        sess = int(fn[0:fn.find('_')])
        try:
            fn_y = [f for f in label_files if fn[0:fn.find('_')] in f][0]
        except:
            continue
        print(fn)
        print(fn_y)
        try:
            X = np.load(os.path.join(args.load_path, fn), allow_pickle=True)
            y = np.load(os.path.join(args.load_path, fn_y), allow_pickle=True)
            print("ok", len(X), len(y))
        except:
            print(fn, "loading failed")
            continue
        try:
            fn_stat = [f for f in stat_files if fn[0:fn.find('_')] in f][0]
            stat = np.load(os.path.join(args.load_path, fn_stat), allow_pickle=True)
            print("loaded stat", len(stat))
            print(np.unique(stat))
        except:
            print("no status data, assuming all full")
            stat = None
        try:
            fn_hq = [f for f in hr_hq_files if fn[0:fn.find('_')] in f][0]
            hq = np.load(os.path.join(dynamic_path, fn_hq), allow_pickle=True)
            print("loaded hq", len(hq))
        except:
            print("no hq data, assuming all hq")
            hq = None

        fn_mean_y = str(sess)+'_ymean.npy'
        np.save(os.path.join(args.save_path, fn_mean_y), np.mean(y[y>0]))

        print(fn, "loaded successfully", X.shape, y.shape)
        idxs = np.array(list(set(X.keys()).intersection(set(y.keys()))))
        print(idxs)
        print(len(idxs))
        for i in idxs:

            fn_X = str(sess)+'_'+str(i)+'_X.npy'
            fn_y = str(sess)+'_'+str(i)+'_y.npy'

            print(i, fn_X ,fn_y )
            np.save(os.path.join(args.save_path, fn_X), X[i], allow_pickle=True)
            st = stat[i] if stat is not None else 'full'
            q = hq[i] if hq is not None and i < len(hq) else 1
            np.save(os.path.join(args.save_path, fn_y), [y[i], st, q], allow_pickle=True)




