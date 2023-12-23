# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
#import stumpy
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
from Tests import vsms_db_api as db_api
from Tests.vsms_db_api import Sensor
from Tests.Utils.LoadingAPI import load_reference, load_phase
import pandas as pd
import matplotlib.pyplot as plt
db = db_api.DB()

delays = {109872: 4, 109884: 1786, 109886: 1806, 109887: 2041, 109892: 2041, 109897: 2181, 109901: 2200, 109903: 5194, 109906: 5230, 109918: 1416, 110071: 1801, 110072: 1806, 110190: 6, 110191: 6, 110452: 6, 110454: 9}



MB_HQ = [109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897
, 109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966, 110033
, 110044, 110045, 110071, 110072, 110190, 110191, 110323, 110331, 110332, 110334
, 110337, 110338, 110340, 110341, 110342, 110343, 110344, 110347, 110348, 110361
, 110362, 110364, 110366, 110368, 110371, 110372, 110376, 110377, 110378, 110379
, 110382, 110389, 110393, 110394, 110396, 110398, 110399, 110400, 110401, 110402
, 110405, 110407, 110408, 110410, 110411, 110412, 110413, 110452, 110454]



def get_signal_time_index(setup, db, periods):
    """ periods is the signal size, ie phase or cpx size/len """
    ts = db.setup_ts(setup, sensor=Sensor.nes)
    time_index = pd.date_range(start=ts['start'], periods=periods, freq='100L').values
    return time_index


def radar_cpx_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=db_api.Sensor.nes)
    dir_radar = os.path.dirname(tlog_files_radar[0])
    return glob.glob(os.path.join(dir_radar, "*.npy"))[0]


def load_radar_data_bin(filename: str, bin=None) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    if bin is None or bin<0 or bin >= len(X):
        x = X[:, np.argmax(np.var(X, axis=0))]
    else:
        x = X[:,bin]
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


def compute_respiration(phase, lp=0.05, hp=10) -> np.ndarray:
    """Filter the phase to get the respiration signal."""
    sos = signal.butter(
        2, [lp, hp], btype="bp", output="sos", fs=500.0
    )

    resp = signal.sosfiltfilt(sos, phase)
    return qnormalize(resp)

def getWakeSegments(setup:int):
    """Compute wake segments per setup, return segments and binary wake/sleep mask"""
    ss_ref = None
    if setup in db.setup_nwh_benchmark():
        ss_ref = load_reference(setup, 'sleep_stages', db)
    else:
        session = db.session_from_setup(setup)
        setups = db.setups_by_session(session)
        for s in setups:
            if s in db.setup_nwh_benchmark():
                ss_ref = load_reference(s, 'sleep_stages', db)
    if ss_ref is None:
        return [None,None]

    if isinstance(ss_ref, pd.core.series.Series):
        ss_ref = ss_ref.to_numpy()

    sleep_time = np.zeros(len(ss_ref))

    sleep_time[ss_ref == None] = 1
    sleep_time[ss_ref == 'W'] = 1

    sleep_diff = np.diff(sleep_time, prepend=0, append=0)

    sleep_changes = np.where(sleep_diff)[0]
    wake_start_idx = sleep_changes[::2]
    wake_end_idx = sleep_changes[1::2]
    wake_duration = wake_end_idx-wake_start_idx

    min_wake_minutes = 2
    wake_segments = []
    for i, w in enumerate(wake_duration):
        if i == 0 or i == len(wake_duration)-1:
            wake_segments.append([wake_start_idx[i], wake_end_idx[i]])
        elif w > min_wake_minutes*60:
            wake_segments.append([wake_start_idx[i], wake_end_idx[i]])

    print(wake_segments)
    updated_wake_mask = np.ones_like(sleep_time)
    for s in wake_segments:
        updated_wake_mask[s[0]:s[1]] = 0

    return [wake_segments, updated_wake_mask]

def getApneaSegments(setup:int, respiration: np.ndarray, fs_new: float):
    """compute apnea segments per session"""
    apnea_ref = None
    db.update_mysql_db(setup)
    if db.mysql_db  == 'neteera_cloud_mirror':
        apnea_ref = load_reference(setup, 'apnea', db)
    else:
        if setup in db.setup_nwh_benchmark():
            apnea_ref = load_reference(setup, 'apnea', db)
        else:
            session = db.session_from_setup(setup)
            setups = db.setups_by_session(session)
            for s in setups:
                if s in db.setup_nwh_benchmark():
                    apnea_ref = load_reference(s, 'apnea', db)

    #print(sess, type(apnea_ref))
    if apnea_ref is None:
        return None

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()
    apnea_from_displacement = np.zeros(len(apnea_ref) * fs_new)

    apnea_segments = []

    for i in range(len(apnea_ref) - 1):
        if apnea_ref[i] not in apnea_class.keys():
            apnea_from_displacement[i * fs_new:(i + 1) * fs_new] = -1
        else:
            apnea_from_displacement[i * fs_new:(i + 1) * fs_new] = apnea_class[apnea_ref[i]]

    if apnea_from_displacement[0] == -1:
        apnea_diff = np.diff(apnea_from_displacement, prepend=-1)
    else:
        apnea_diff = np.diff(apnea_from_displacement, prepend=0)

    apnea_changes = np.where(apnea_diff)[0]
    if apnea_from_displacement[0] == -1:
        apnea_changes = apnea_changes[1:]
    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]
    apnea_type = apnea_from_displacement[apnea_idx]  # apneas[:, 2]

    apneas_mask = np.zeros(len(respiration))

    for a_idx, start_idx in enumerate(apnea_idx):
        if apnea_type[a_idx] not in apnea_class.values():
            print(apnea_type[a_idx], "not in apnea_class.values()")
            continue
        if float(apnea_duration[a_idx]) == 0.0:
            continue
        end_idx = apnea_end_idx[a_idx]
        apneas_mask[start_idx:end_idx] = 1

        apnea_segments.append([start_idx, end_idx, apnea_duration, apnea_type[a_idx]])
    return apnea_segments


def getSetupRespirationLocalDBDebug(sess: int, bin_index=None):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dir = os.sep.join([p, 'NES_RAW'])

    radar_file = radar_cpx_file(sess)

    cpx, setup_fs = load_radar_data_bin(radar_file, bin_index)
    phase = compute_phase(cpx)

    bins = None
    phase_df = pd.DataFrame(phase)
    i = cpx.real
    q = cpx.imag
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    respiration = compute_respiration(ph.flatten())
    UP = 1
    DOWN = 50
    respiration = sp.resample_poly(respiration, UP, DOWN)

    I = sp.resample_poly(i_np, UP, DOWN)
    Q = sp.resample_poly(q_np, UP, DOWN)

    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins, I, Q


if __name__ == '__main__':

    # args = get_args()

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']
    home = '/Neteera/Work/homes/dana.shavit/'

    setups = [9002, 9003, 9187, 8998]

    print(setups)
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        for bi in range(12):
            fig, ax = plt.subplots(3, sharex=True)
            respiration, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(sess, bi)

            # apnea_segments = getApneaSegments(sess, respiration, fs_new)
            # for s in apnea_segments:
            #     ax[0].axvspan(s[0], s[1], color='yellow', alpha=0.3)
            #     ax[3].axvspan(s[0], s[1], color='yellow', alpha=0.3)
            ax[0].set_title('bin'+str(bi)+' '+str(sess) + ' ' + db.setup_subject(sess) + ' ' + db.setup_sn(sess)[0])

            ax[0].plot(respiration, label='displacement')
            ax[1].plot(I, label='I')
            ax[2].plot(Q, label='Q')
            if sess > 9200:
                # p = db.setup_dir(sess)
                # raw_dir = os.sep.join([p, 'NES_RAW'])
                # res_dir = os.sep.join([p, 'NES_RES'])
                #
                # onlyfiles = os.listdir(res_dir)
                # vs = [f for f in onlyfiles if 'VS' in f and 'csv' in f]
                #
                # if len(onlyfiles) == 1:
                #     res_dir = os.sep.join([res_dir, onlyfiles[0]])
                #     onlyfiles = os.listdir(res_dir)
                #     vs = [f for f in onlyfiles if 'results' in f and 'csv' in f]

                # vs = vs[0]
                # df = pd.read_csv(os.sep.join([res_dir, vs]))
                # rr = np.repeat(df['rr'].to_numpy(), fs_new)
                # rr_hq = df['rr_hq'].to_numpy()
                # hr = np.repeat(df['hr'].to_numpy(), fs_new)
                # hr_hq = df['hr_hq'].to_numpy()
                # status = df['stat']


                #ax[1].plot(rr, label='rr')
                #ax[2].plot(hr, label='hr')

                ax[1].legend()
                ax[1].axhline(y=10, color='magenta')
                ax[2].legend()
                # ax[3].legend()
                ax[0].legend()

                empty = df['stat'] == 'Empty'
                running = df['stat'] == 'Running'
                lowrr = df['stat'] == 'Low-respiration'
                motion = df['stat'] == 'Motion'

                empty[empty == True] = 1
                empty[empty == False] = 0

                empty = empty.to_numpy()

                lowrr[lowrr == True] = 1
                lowrr[lowrr == False] = 0
                lowrr = lowrr.to_numpy()

                motion[motion == True] = 1
                motion[motion == False] = 0
                motion = motion.to_numpy()

            plt.savefig(home + str(sess) + '_' +str(bi)+'_'+ db.setup_sn(sess)[0] + '_'+str(bi)+ '.png', dpi=300)
            plt.show()
            plt.close()
            d = 10000
            # for i in range(d, len(I), d):
            #     I_seg = I[i - d:i]
            #     Q_seg = Q[i - d:i]
            #     plt.figure()
            #     plt.plot(I_seg)
            #     plt.plot(Q_seg)
            #     plt.title(str(sess) + '_' + str(i - d) + ":" + str(i))
            #     nf = os.path.join(home, str(sess) + '_' + str(bi)+"_"+str(i - d) + "-" + str(i) + '_seg.png')
            #     plt.savefig(nf, dpi=1200)
            #     plt.show()
            #     plt.close()
            # continue


