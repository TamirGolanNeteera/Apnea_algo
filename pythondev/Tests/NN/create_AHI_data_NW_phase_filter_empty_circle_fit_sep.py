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

from create_apnea_count_AHI_data import MB_HQ, create_AHI_training_data, create_AHI_training_data_no_wake_no_empty, count_apneas_in_chunk, getApneaSegments, getWakeSegments
from create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels

db = db_api.DB()
logger = logging.getLogger(__name__)
# db = DB()

def radar_cpx_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=db_api.Sensor.nes)
    dir_radar = os.path.dirname(tlog_files_radar[0])

    return glob.glob(os.path.join(dir_radar, "*.npy"))[0]


def load_radar_data(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    x = X[:, np.argmax(np.var(X, axis=0))]
    return x, 1 / Ts


def circlecenter(x, y):
    """Given points on a circle with coordinates (x,y), compute the center of the circle."""
    f = 10  # controls numerical accuracy. Some number same-ish order of magnitude as I and Q.
    Nx = len(x)

    # Correlation of data
    Mxy = np.vstack(
        (
            np.square(x) + np.square(y),
            2 * f * x,
            2 * f * y,
            f**2 * np.ones(Nx),
        )
    )
    mn = np.mean(Mxy, axis=1)
    Cmx = np.cov(Mxy)
    X = Cmx + np.outer(mn, mn)

    xc = np.mean(x)
    yc = np.mean(y)
    sxs = np.var(x)
    sys = np.var(y)
    sx = sxs + xc**2
    sy = sys + yc**2
    N = np.vstack(
        (
            [4 * (sx + sy), (2 * f * xc), (2 * f * yc), 0],
            [(2 * f * xc), (f**2), 0, 0],
            [(2 * f * yc), 0, (f**2), 0],
            [0, 0, 0, 0],
        )
    )
    D, V = linalg.eig(X, N)
    i = np.argsort(np.abs(D))
    v = V[:, i[0]]
    if v[0] != 0.0:
        v = v / v[0]

    xch = -v[1] * f / v[0]
    ych = -v[2] * f / v[0]

    return xch, ych


def compute_phase(iq: np.ndarray) -> np.ndarray:
    """Return the phase of the radar signal after removing the bias."""
    cr, ci = circlecenter(np.real(iq), np.imag(iq))
    return np.unwrap(np.angle(iq - (cr + 1j * ci)))


def qnormalize(x, lo=0.1, hi=0.9):
    """Squeeze the signal x to the range (-1,1) using quantiles."""
    qlo, qhi = np.quantile(x, lo), np.quantile(x, hi)
    return 2 * (x - qlo) / (qhi - qlo) - 1


def compute_respiration(phase: np.ndarray) -> np.ndarray:
    """Filter the phase to get the respiration signal."""
    sos = signal.butter(
        2, [0.05, 10], btype="bp", output="sos", fs=500.0
    )

    resp = signal.sosfiltfilt(sos, phase)
    return qnormalize(resp)


def getSetupRespiration(sess: int):
    radar_file = radar_cpx_file(sess)
    iq_data, setup_fs = load_radar_data(radar_file)

    phase = compute_phase(iq_data)
    respiration = compute_respiration(phase)
    UP = 1
    DOWN = 50
    respiration = sp.resample_poly(respiration, UP, DOWN)
    setup_fs = int((db.setup_fs(sess) * UP) / DOWN)
    return respiration, setup_fs


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

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    sessions = []
    nwh_company = db.setup_by_project(db_api.Project.nwh)
    bed_top = db.setup_by_mount(db_api.Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=db_api.Sensor.nes, value=db_api.Validation.invalid)
    #setups = set(bed_top) & set(nwh_company) - set(invalid) - {6283, 9096, 8734, 8705}

    setups = set(nwh_company) - set(invalid)
    for s in setups:
        natus_validity = db.setup_data_validity(s, db_api.Sensor.natus)
        if natus_validity in ['Confirmed', 'Valid']:
            sessions.append(s)


    """use to display signal then exit"""
    if args.show:
        sessions = [8724, 6284, 8674, 8995, 8580, 6284, 9053, 9100]
    else:
        sessions = [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735,
                 8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]
    """Iterate over all sessions to create data"""


    for i_sess, sess in enumerate(sessions):
        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(sessions)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path,str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue
        respiration, fs_new = getSetupRespiration(sess)
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
        chunk_size_in_minutes = args.chunk
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
