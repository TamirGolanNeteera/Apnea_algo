# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse

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

import numpy as np
# import dbinterface
from Tests import vsms_db_api as db_api

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
 parser.add_argument('--hypopneas', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')
 parser.add_argument('--show', action='store_true', required=False, help='display session only')
 parser.add_argument('-down', metavar='window', type=int, required=False, help='downsample')
 parser.add_argument('-lp', metavar='window', type=float, required=False, help='lowpass')
 parser.add_argument('-hp', metavar='window', type=float, required=False, help='highpass')
 return parser.parse_args()

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

    lp = 0.05 if args.lp is None else args.lp
    hp = 10 if args.lp is None else args.hp
    down = 50 if args.down is None else args.down

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    sessions = []
    nwh_company = db.setup_by_project(db_api.Project.nwh)
    bed_top = db.setup_by_mount(db_api.Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=db_api.Sensor.nes, value=db_api.Validation.invalid)
    setups = set(bed_top) & set(nwh_company) - set(invalid) - {6283, 9096,  8705}

    #setups = set(nwh_company) - set(invalid)- {6283, 9096, 8705}
    for s in setups:
        natus_validity = db.setup_data_validity(s, db_api.Sensor.natus)
        if natus_validity in ['Confirmed', 'Valid']:
            sessions.append(s)


    """use to display signal then exit"""
    if args.show:
        setups = [9187, 9188]
    else:
        setups = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735, 8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]
    """Iterate over all sessions to create data"""


    for i_sess, sess in enumerate(setups):
        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(setups)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path,str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue
        respiration, fs_new = getSetupRespiration(sess, down, lp, hp)
        apnea_segments = getApneaSegments(sess, respiration, fs_new)
        if not apnea_segments:
            print("ref not ok")
            continue
        else:
            print(sess, len(apnea_segments), "apneas in segment")

        if args.show:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1)
            ax.plot(respiration)
            for s in apnea_segments:
                c = 'red' if s[3] != 2.0 else 'green'
                ax.axvspan(s[0], s[1], color=c, alpha=0.3)
                plt.title(str(sess))

            plt.show()
            continue


        X = []
        y = []
        chunk_size_in_minutes = args.chunk1
        time_chunk = fs_new * chunk_size_in_minutes * 60
        step = fs_new * args.step * 60

        """use wake to filter out segments with > 50% wake seconds for cleaner trainig set"""


        ss_ref_class = get_sleep_stage_labels(sess)
        if len(ss_ref_class) == 0:
            print('ref bad')
            continue
        empty = get_empty_seconds_nw(sess)
        X, y, valid = create_AHI_training_data_no_wake_no_empty(respiration=respiration, apnea_segments=apnea_segments, wake_ref=ss_ref_class, empty_ref=empty, time_chunk=time_chunk, step=step, scale=args.scale, fs=fs_new)


        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_valid.npy'), valid, allow_pickle=True)

        print("saved training data")
