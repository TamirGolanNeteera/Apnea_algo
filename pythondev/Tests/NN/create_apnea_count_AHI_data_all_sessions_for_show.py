# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential

#from Tests.Plots.PlotRawDataRadarCPX import*
import argparse

import scipy.signal as sp
from scipy import linalg, signal
import glob
import logging
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from create_apnea_count_AHI_data import create_AHI_training_data, create_AHI_training_data_no_wake, count_apneas_in_chunk, getApneaSegments, getWakeSegments
from create_apnea_count_AHI_data_segmentation_filter_wake_option import get_sleep_stage_labels
from Tests.Utils.DBUtils import find_back_setup_same_session, match_lists_by_ts
from Tests.Utils.LoadingAPI import load_nes, load_reference
from Tests.vsms_db_api import *

db =DB()

def radar_cpx_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=Sensor.nes)
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
        2, [0.05, 3.3333], btype="bp", output="sos", fs=500.0
    )

    resp = signal.sosfiltfilt(sos, phase)
    return qnormalize(resp)


def getSetupRespiration(sess: int):
    radar_file = radar_cpx_file(sess)
    iq_data, setup_fs = load_radar_data(radar_file)
    if sess == 8705:
        iq_data = iq_data[:15000000]

    phase = compute_phase(iq_data)
    respiration = compute_respiration(phase)
    UP = 1
    DOWN = 10
    respiration = sp.resample_poly(respiration, UP, DOWN)
    setup_fs = int((db.setup_fs(sess) * UP) / DOWN)
    return respiration, setup_fs


def show_posture(sess: int):
    # radar_file = radar_cpx_file(sess)
    # iq_data, setup_fs = load_radar_data(radar_file)
    # UP = 1
    # DOWN = 10
    # iq = sp.resample_poly(iq_data, UP, DOWN)
    #
    # i = iq.real
    # q = iq.imag
    #
    # sos = signal.butter(
    #     2, [0.05, 3.3333], btype="bp", output="sos", fs=500.0
    # )
    # i = signal.sosfiltfilt(sos, i)
    # q = signal.sosfiltfilt(sos, q)

    db.update_mysql_db(sess)

    respiration, setup_fs = getSetupRespiration(sess)
    if sess in db.setup_nwh_benchmark():
        posture = load_reference(sess, 'posture', db)
        if posture is None:
            return
    else:
        session = db.session_from_setup(sess)
        setups = db.setups_by_session(session)
        for s in setups:
            if s in db.setup_nwh_benchmark():
                posture = load_reference(s, 'posture', db)
                if posture is None:
                    return


    posture = posture.to_numpy()

    postures_dict = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}

    for ip, p in enumerate(posture):
        if p == None:
            posture[ip] = -1
        elif p in postures_dict.keys():
            posture[ip] = postures_dict[p]
        else:
            posture[ip] = -1

    posture[0] = -2
    posture_diff = np.diff(posture, prepend=-1)
    posture_changes = np.where(posture_diff)[0]

    fig, ax = plt.subplots(2, sharex=False, gridspec_kw={'hspace': 0})
    ax[0].plot(respiration, alpha=0.5, color='black')
    ax[1].plot(respiration, alpha=0.5, color='black')
    col = ['gray', 'blue', 'green', 'cyan', 'magenta', 'red', 'orange', 'gray', 'yellow', 'salmon', 'gold', 'darkgreen']
    for i in range(len(posture_changes) - 1):
        print(posture_changes[i], posture_changes[i+1], posture[posture_changes[i]], posture[posture_changes[i]+1], col[posture[posture_changes[i]+1]])
        ax[0].axvspan(50*posture_changes[i], setup_fs*posture_changes[i + 1], color=col[posture[posture_changes[i]+1]], alpha=0.3)
        ax[1].axvspan(50*posture_changes[i], setup_fs*posture_changes[i + 1], color=col[posture[posture_changes[i]+1] ], alpha=0.3)
    #print(posture[1:1000])
    plt.show()

def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')

 parser.add_argument('-session', metavar='window', type=int, required=True, help='sesison number to show')

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


    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    sessions = []
    nwh_company = db.setup_by_project(Project.nwh)
    bed_top = db.setup_by_mount(Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
    setups = db.setup_nwh_benchmark()#set(bed_top) & set(nwh_company) - set(invalid) - {6283, 9096, 8734, 8705}

    #setups = set(nwh_company) - set(invalid) - {8705, 8710}
    for s in setups:
        natus_validity = db.setup_data_validity(s, Sensor.natus)
        if natus_validity in ['Confirmed', 'Valid']:
            sessions.append(s)
    bm_sessions = db.setup_nwh_benchmark()
    print(bm_sessions)

    """use to display signal then exit"""
    sessions = [int(args.session)]
    main_session = db.session_from_setup(args.session)
    radar_setups = db.setups_by_session(main_session)
    """Iterate over all sessions to create data"""
    for i_sess, sess in enumerate(sessions):
        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(sessions)), "::::::::")

        #show_posture(sess)
        #continue
        db.update_mysql_db(sess)
        data = load_nes(sess, db)['data']
        respiration, fs_new = getSetupRespiration(sess)
        apnea_segments = getApneaSegments(sess, respiration, fs_new)
        if not apnea_segments:
            print("ref not ok")
            continue
        wake_segments = getWakeSegments(sess)
        db.update_mysql_db(sess)
        reference = load_reference(sess, 'chest', db)

        chest = reference.to_numpy()
        UP = 25
        DOWN = 256
        chest[chest == None] = 0
        chest = sp.resample_poly(chest, UP, DOWN)
        #plt.plot(chest)

        fig, ax = plt.subplots(2,sharex=True, gridspec_kw={'hspace': 0})
        ax[0].plot(respiration)

        ax[1].plot(chest)
        #ax[1].plot(respiration)
        for s in apnea_segments:
            c = 'red' if s[3] != 2.0 else 'green'
            ax[0].axvspan(s[0], s[1], color=c, alpha=0.3)
            ax[1].axvspan(s[0], s[1], color=c, alpha=0.3)
            plt.title(str(sess))
        for s in wake_segments[0]:
            c = 'yellow'
            ax[0].axvspan(fs_new*s[0], fs_new*s[1], color=c, alpha=0.3)
            ax[1].axvspan(fs_new*s[0], fs_new*s[1], color=c, alpha=0.3)

        plt.show()
        continue
