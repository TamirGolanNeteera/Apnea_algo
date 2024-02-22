# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import stumpy
from matplotlib.patches import Rectangle
from sklearn import preprocessing
import random
from statistics import mode
import scipy.signal as sp
from scipy import linalg, signal
import glob
import logging
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import dbinterface
from Tests import vsms_db_api as db_api
from Tests.Utils.LoadingAPI import load_reference
from create_apnea_count_AHI_data import MB_HQ, create_AHI_training_data, create_AHI_training_data_no_wake_no_empty, count_apneas_in_chunk, getApneaSegments, getWakeSegments, getSetupRespiration
from create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels

db = db_api.DB()
logger = logging.getLogger(__name__)

def get_empty_seconds_nw(setup):
    base_path = '/Neteera/Work/NeteeraVirtualServer/DELIVERED/Algo/net-alg-3.5.9/stats/nwh/dynamic'
    stat_fn = str(setup)+'_stat.data'

    try:
        stat = np.load(os.sep.join([base_path, stat_fn]), allow_pickle=True)
    except:
        return None
    empty = np.ones(len(stat))
    for i in range(len(stat)):
        if stat[i] == 'empty chair' or stat == 'warming up':
            empty[i] = 0


    return empty

def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 parser.add_argument('-save_path', metavar='Location', type=str, required=True, help='location of saved data')
 parser.add_argument('--scale', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--filter_empty', action='store_true', help='filter empty data from input')
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')
 parser.add_argument('--show', action='store_true', required=False, help='display session only')

 return parser.parse_args()


def create_AHI_regression_training_data_NW_phase(respiration, apnea_ref, empty_seconds, time_chunk, step, scale, fs):
    print("in")
    X = []
    y = []

    valid = []
    apnea_segments = []

    #empty_seconds = [max(empty_seconds[i], empty_bins[i]) for i in range(len(empty_seconds))]

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()

    apnea_diff = np.diff(apnea_ref, prepend=0)

    apnea_changes = np.where(apnea_diff)[0]
    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]
    apnea_type = 1  # apneas[:, 2]

    for a_idx, start_idx in enumerate(apnea_idx):

        if float(apnea_duration[a_idx]) == 0.0:
            continue
        end_idx = apnea_end_idx[a_idx]

        apnea_segments.append([start_idx, end_idx, apnea_duration, 1    ])
    print(len(apnea_segments), "apneas in setup")

    filter_hour_1 = True
    delay = 60 if filter_hour_1 else 0
    print("delay = ", delay)
    for i in range(time_chunk, len(respiration), step):
        # for i in range(time_chunk, len(respiration), step):
        v = 1

        if i < int(delay * fs):
            v = 0
        seg = respiration[i - time_chunk:i]
        start_fs = int((i - time_chunk)/fs)
        end_fs = int(i/fs)
        if empty_seconds is not None and end_fs < len(empty_seconds):
            empty_ref = empty_seconds[start_fs:end_fs]

            if np.sum(empty_ref)/len(empty_ref) > 0.3:
                v = 0
            #print(start_fs, end_fs, np.sum(empty_ref), len(empty_ref), v)
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
        if len(seg) != time_chunk:
            continue


        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=start_fs, end_t=end_fs, apnea_segments=apnea_segments)


        y.append(num_apneas)

        valid.append(v)

    if len(X):
        X = np.stack(X)
        y = np.stack(y)


        valid = np.stack(valid)
        print(np.count_nonzero(valid))
        print(len(X), len(y), np.sum(valid) / len(valid))
        return X, y, valid


    return X,y, valid
#


def create_AHI_regression_training_data_NW_phase_for_augmentation(respiration, apnea_ref, empty_seconds, time_chunk, step, scale, fs):
    print("in")
    X = []
    y = []
    y_3 = []
    valid = []
    apnea_segments = []

    # empty_seconds = [max(empty_seconds[i], empty_bins[i]) for i in range(len(empty_seconds))]

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()

    apnea_diff = np.diff(apnea_ref, prepend=0)

    apnea_changes = np.where(apnea_diff)[0]
    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]
    apnea_type = 1  # apneas[:, 2]

    for a_idx, start_idx in enumerate(apnea_idx):

        if float(apnea_duration[a_idx]) == 0.0:
            continue
        end_idx = apnea_end_idx[a_idx]

        apnea_segments.append([start_idx, end_idx, apnea_duration, 1])
    print(len(apnea_segments), "apneas in setup")

    for i in range(time_chunk, len(respiration), step):

        v = 1
        seg = respiration[i - time_chunk:i]
        start_fs = int((i - time_chunk) / fs)
        end_fs = int(i / fs)
        if empty_seconds is not None and end_fs < len(empty_seconds):
            empty_ref = empty_seconds[start_fs:end_fs]

            if np.sum(empty_ref) / len(empty_ref) > 0.3:
                v = 0
            # print(start_fs, end_fs, np.sum(empty_ref), len(empty_ref), v)
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
        if len(seg) != time_chunk:
            continue

        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=start_fs, end_t=end_fs, apnea_segments=apnea_segments)

        third = int(time_chunk / 3)
        y3 = []
        for j in range(3):
            ss = start_fs + j * third
            es = start_fs + (j + 1) * third
            num_apneas_in_third = count_apneas_in_chunk(start_t=ss, end_t=es, apnea_segments=apnea_segments)
            y3.append(num_apneas_in_third)
        y.append(num_apneas)
        y_3.append(y3)
        valid.append(v)

    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        y_3 = np.stack(y_3)

        valid = np.stack(valid)
        print(np.count_nonzero(valid))
        print(len(X), len(y), np.sum(valid) / len(valid))
        return X, y, valid, y_3

    return X, y, valid, y_3
"""should be moved to common enums resource file"""
apnea_class = {'missing': -1,
        'Normal Breathing': 0,
        'normal': 0,
        'Central Apnea': 1,
        'Hypopnea': 2,
        'Mixed Apnea': 3,
        'Obstructive Apnea': 4,
        'Noise': 5}


if __name__ == '__main__':

    args = get_args()

    if args.scale:
        save_path = os.path.join(args.save_path, 'scaled')
    else:
        save_path = os.path.join(args.save_path, 'unscaled')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    """use to display signal then exit"""
    setups = [8674, 8995, 8580, 9093, 8998, 9188, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735, 8710, 8584, 8994, 8579, 9094, 8999, 9187, 9101, 8724, 8756, 8919, 6641, 9050]
    """Iterate over all sessions to create data"""


    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

        print(":::::::: processing session", sess, str(i_sess) + '/' + str(len(setups)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path, str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue
        try:
            if sess in db.setup_nwh_benchmark():
                apnea_reference = load_reference(sess, 'apnea', db)
            else:
                session = db.session_from_setup(sess)
                setups = db.setups_by_session(session)
                for s in setups:
                    if s in db.setup_nwh_benchmark():
                        apnea_reference = load_reference(s, 'apnea', db)
            #apnea_reference = load_apnea_ref(sess, db)
            # respiration, fs_new = getSetupRespirationCloudDB(sess)
            if apnea_reference is None:
                print("no reference for setup", sess)

            if isinstance(apnea_reference, pd.core.series.Series):
                apnea_reference = apnea_reference.to_numpy()


            apnea_ref_class = np.zeros(len(apnea_reference))

            for i in range(len(apnea_reference)):
                # print(i, apnea_reference[i], apnea_reference[i] in apnea_class.keys())
                if apnea_reference[i] not in apnea_class.keys():
                    apnea_ref_class[i] = -1
                else:
                    apnea_ref_class[i] = int(apnea_class[apnea_reference[i]])

            print(np.unique(apnea_ref_class))

            respiration, fs_new = getSetupRespiration(sess)



            min_setup_length = 15000
            if len(respiration) / fs_new < min_setup_length:
                continue

            chunk_size_in_minutes = args.chunk
            time_chunk = fs_new * chunk_size_in_minutes * 60
            step = fs_new * args.step * 60

            empty_ref_nw = None
            if args.filter_empty:
                empty_ref_nw = get_empty_seconds_nw(sess)

            X, y, valid = create_AHI_regression_training_data_NW_phase(respiration=respiration,
                                                                   apnea_ref=apnea_ref_class,
                                                                   empty_seconds=empty_ref_nw,
                                                                   # bins_ref=empty_bins,
                                                                   time_chunk=time_chunk,
                                                                   step=step,
                                                                   scale=args.scale,
                                                                   fs=fs_new)
        except:
            print("not ok")
            continue

        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        #np.save(os.path.join(save_path,str(sess) + '_y3.npy'), y3, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_valid.npy'), valid, allow_pickle=True)

        print("saved training data")
