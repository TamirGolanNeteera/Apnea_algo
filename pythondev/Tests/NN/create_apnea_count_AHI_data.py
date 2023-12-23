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

NW_HQ =  [8674, 8995, 8580, 9093, 8998, 9188, 9003, 6284, 9100, 8719, 8757, 8920, 6651, 9053, 8735, 8710, 8584, 8994, 8579, 9094, 8999, 9187, 9002, 9101, 8724, 8756, 8919, 6641, 9050]


apnea_class = {'missing': -1,
        'Normal Breathing': 0,
        'normal': 0,
        'Central Apnea': 1,
        'Central': 1,
        'Hypopnea': 1,
        'Obstructive Hypopnea': 1,
        'Mixed Apnea': 1,
        'Mixed': 1,
        'Apnea': 1,
        'Obstructive Apnea': 1,
        'Noise': -1}

def get_signal_time_index(setup, db, periods):
    """ periods is the signal size, ie phase or cpx size/len """
    ts = db.setup_ts(setup, sensor=Sensor.nes)
    time_index = pd.date_range(start=ts['start'], periods=periods, freq='500L').values#500L=500Hz
    return time_index


def radar_cpx_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=db_api.Sensor.nes)
    dir_radar = os.path.dirname(tlog_files_radar[0])
    return glob.glob(os.path.join(dir_radar, "*.npy"))[0]


def load_radar_data(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    x = X[:, np.argmax(np.var(X, axis=0))]
    print("maxbin = ",  np.argmax(np.var(X, axis=0)))

    return x, 1 / Ts\

def load_radar_bins(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    return X


def load_radar_data_top3bins(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    x = X[:, np.argsort(np.var(X, axis=0))[-3:]]
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


def compute_hr(phase, lp=10, hp=20) -> np.ndarray:
    """Filter the phase to get the respiration signal."""
    sos = signal.butter(
        2, [lp, hp], btype="bp", output="sos", fs=500.0
    )

    resp = signal.sosfiltfilt(sos, phase)
    return qnormalize(resp)

def getSetupRespiration(sess, down=50, lp=0.05, hp=10):
    """return respiration displacement signal for setup"""
    radar_file = radar_cpx_file(sess)
    iq_data, setup_fs = load_radar_data(radar_file)

    phase = compute_phase(iq_data)
    respiration = compute_respiration(phase, lp, hp)
    UP = 1
    DOWN = down
    respiration = sp.resample_poly(respiration, UP, DOWN)
    setup_fs = int((db.setup_fs(sess) * UP) / DOWN)
    return respiration, setup_fs


def getSetupRespirationCloudDB(sess, down=50, lp=0.05, hp=10):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dir = os.sep.join([p, 'NES_RAW'])
    for file in os.listdir(raw_dir):
        if 'phase.npy' in file and 'tmp' not in file:
            phase_fn = file
            phase_path = os.path.join(raw_dir, phase_fn)
            break
    for file in os.listdir(raw_dir):
        if 'bins.npy' in file and 'tmp' not in file:
            bins_fn = file
            bins_path = os.path.join(raw_dir, bins_fn)
            break

    phase = np.load(os.sep.join([raw_dir, phase_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dir, bins_fn]), allow_pickle=True)
    phase_df = pd.DataFrame(phase)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    respiration = compute_respiration(ph.flatten(), lp, hp)
    UP = 1
    DOWN = down
    respiration = sp.resample_poly(respiration, UP, DOWN)
    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins


def getSetupRespirationLocalDBReversePreprocessing(sess):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    radar_file = radar_cpx_file(sess)
    iq_data, setup_fs = load_radar_data(radar_file)

    UP = 1
    DOWN = 50

    I = sp.resample_poly(iq_data.real, UP, DOWN)
    Q = sp.resample_poly(iq_data.imag, UP, DOWN)
    ph = compute_phase(np.squeeze(I+Q*1.0j))
    respiration = compute_respiration(ph.flatten())

    setup_fs = int((db.setup_fs(sess) * UP) / DOWN)
    return respiration, setup_fs



    cpx = np.load(os.sep.join([raw_dirr, cpx_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dirr, bins_fn]), allow_pickle=True)
    i = cpx[:, 0]
    q = cpx[:, 1]
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    #i_df.interpolate(method="linear", inplace=True)
    i_df.fillna(method="bfill", inplace=True)
    i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    q_df.fillna(method="bfill", inplace=True)
    q_df.fillna(method="pad", inplace=True)

    UP = 1
    DOWN = 50
    I = sp.resample_poly(i_df, UP, DOWN)
    Q = sp.resample_poly(q_df, UP, DOWN)

    ph = compute_phase(np.squeeze(I+Q*1.0j))
    respiration = compute_respiration(ph.flatten())

    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins


def getSetupRespirationCloudDBReversePreprocessing(sess):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dirr = os.sep.join([p, 'NES_RAW'])
    cpx_fn = None

    for file in os.listdir(raw_dirr):
        if 'cpx.npy' in file and 'tmp' not in file:
            cpx_fn = file
            cpx_path = os.path.join(raw_dirr, cpx_fn)
            break
    for file in os.listdir(raw_dirr):
        if 'bins.npy' in file and 'tmp' not in file:
            bins_fn = file
            bins_path = os.path.join(raw_dirr, bins_fn)
            break

    cpx = np.load(os.sep.join([raw_dirr, cpx_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dirr, bins_fn]), allow_pickle=True)
    i = cpx[:, 0]
    q = cpx[:, 1]
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    #i_df.interpolate(method="linear", inplace=True)
    i_df.fillna(method="bfill", inplace=True)
    i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    q_df.fillna(method="bfill", inplace=True)
    q_df.fillna(method="pad", inplace=True)

    UP = 1
    DOWN = 50
    I = sp.resample_poly(i_df, UP, DOWN)
    Q = sp.resample_poly(q_df, UP, DOWN)

    ph = compute_phase(np.squeeze(I+Q*1.0j))
    respiration = compute_respiration(ph.flatten())

    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins



def getSetupRespirationCloudDBDebug(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dir = os.sep.join([p, 'NES_RAW'])
    print(os.listdir(raw_dir))
    for file in os.listdir(raw_dir):
        if 'phase.npy' in file and 'tmp' not in file:
            phase_fn = file
            phase_path = os.path.join(raw_dir, phase_fn)
            break
    for file in os.listdir(raw_dir):
        if 'cpx.npy' in file and 'tmp' not in file:
            cpx_fn = file
            cpx_path = os.path.join(raw_dir, cpx_fn)
            break
    for file in os.listdir(raw_dir):
        if 'bins.npy' in file and 'tmp' not in file:
            bins_fn = file
            bins_path = os.path.join(raw_dir, bins_fn)
            break

    phase = np.load(os.sep.join([raw_dir, phase_fn]), allow_pickle=True)
    cpx = np.load(os.sep.join([raw_dir, cpx_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dir, bins_fn]), allow_pickle=True)


    phase_df = pd.DataFrame(phase)
    i = cpx[:,0]
    q = cpx[:,1]
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    #i_df.interpolate(method="linear", inplace=True)
    #i_df.fillna(method="bfill", inplace=True)
    #i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    #q_df.fillna(method="bfill", inplace=True)
    #q_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    respiration = compute_respiration(ph.flatten())
    UP = 1
    DOWN = 1
    respiration = sp.resample_poly(respiration, UP, DOWN)
    #ts = get_signal_time_index(sess, db, len(respiration))
    I = sp.resample_poly(i_np, UP, DOWN)
    Q = sp.resample_poly(q_np, UP, DOWN)

    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins, I, Q
    #return respiration, 500, bins, i, q


def getSetupRespirationCloudDBDebugDecimate(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dir = os.sep.join([p, 'NES_RAW'])
    print(os.listdir(raw_dir))
    for file in os.listdir(raw_dir):
        if 'phase.npy' in file and 'tmp' not in file:
            phase_fn = file
            phase_path = os.path.join(raw_dir, phase_fn)
            break
    for file in os.listdir(raw_dir):
        if 'cpx.npy' in file and 'tmp' not in file:
            cpx_fn = file
            cpx_path = os.path.join(raw_dir, cpx_fn)
            print(cpx_path)
            break
    for file in os.listdir(raw_dir):
        if 'bins.npy' in file and 'tmp' not in file:
            bins_fn = file
            bins_path = os.path.join(raw_dir, bins_fn)
            break

    phase = np.load(os.sep.join([raw_dir, phase_fn]), allow_pickle=True)
    cpx = np.load(os.sep.join([raw_dir, cpx_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dir, bins_fn]), allow_pickle=True)


    phase_df = pd.DataFrame(phase)
    i = cpx[:,0]
    q = cpx[:,1]
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    respiration = compute_respiration(ph.flatten())

    f = 50
    respiration = respiration[::f]
    I = i_np[::f]
    Q = q_np[::f]

    setup_fs = int(500/f)
    return respiration, setup_fs, bins, I, Q
    #return respiration, 500, bins, i, q

def getSetupRespiration_top3bins(sess, down=50, lp=0.05, hp=10):
    """return respiration displacement signal for setup"""
    radar_file = radar_cpx_file(sess)
    iq_data, setup_fs = load_radar_data_top3bins(radar_file)

    UP = 1
    DOWN = down
    phase = np.zeros(iq_data.shape)
    respiration = []
    for i in range(phase.shape[1]):
        phase[:,i] = compute_phase(iq_data[:,i])
        r = compute_respiration(phase[:,i], lp, hp)
        r = sp.resample_poly(r, UP, DOWN)
        respiration.append(r)
    setup_fs = int((db.setup_fs(sess) * UP) / DOWN)
    return np.stack(respiration), setup_fs

def getSetupRespirationCloudDBDebugWithTimestamp(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)


    #plt.plot(time_index, signal)

    p = db.setup_dir(sess)

    raw_dir = os.sep.join([p, 'NES_RAW'])
    for file in os.listdir(raw_dir):
        if 'phase.npy' in file and 'tmp' not in file:
            phase_fn = file
            phase_path = os.path.join(raw_dir, phase_fn)
            break
    for file in os.listdir(raw_dir):
        if 'cpx.npy' in file and 'tmp' not in file:
            cpx_fn = file
            cpx_path = os.path.join(raw_dir, cpx_fn)
            break
    for file in os.listdir(raw_dir):
        if 'bins.npy' in file and 'tmp' not in file:
            bins_fn = file
            bins_path = os.path.join(raw_dir, bins_fn)
            break

    phase = np.load(os.sep.join([raw_dir, phase_fn]), allow_pickle=True)
    cpx = np.load(os.sep.join([raw_dir, cpx_fn]), allow_pickle=True)
    bins = np.load(os.sep.join([raw_dir, bins_fn]), allow_pickle=True)


    phase_df = pd.DataFrame(phase)
    i = cpx[:,0]
    q = cpx[:,1]
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    #i_df.interpolate(method="linear", inplace=True)
    #i_df.fillna(method="bfill", inplace=True)
    #i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    #q_df.fillna(method="bfill", inplace=True)
    #q_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    respiration = compute_respiration(ph.flatten())
    UP = 1
    DOWN = 50
    respiration = sp.resample_poly(respiration, UP, DOWN)
    ts = get_signal_time_index(sess, db, len(respiration))
    I = sp.resample_poly(i_np, UP, DOWN)
    Q = sp.resample_poly(q_np, UP, DOWN)

    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs, bins, I, Q, ts


def getSetupRespirationLocalDBDebug(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)

    if sess > 9999:
        p = os.sep.join([p, 'NES_RAW/NES/'])

    radar_file = radar_cpx_file(sess)
    cpx, setup_fs = load_radar_data(radar_file)

    phase = compute_phase(cpx)
    #np.save('/Neteera/Work/homes/dana.shavit/phase.npy', phase, allow_pickle=True)

    bins = None
    phase_df = pd.DataFrame(phase)
    i = cpx.real
    q = cpx.imag
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    bins = load_radar_bins(radar_file)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    #i_df.interpolate(method="linear", inplace=True)
    #i_df.fillna(method="bfill", inplace=True)
    #i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    #q_df.fillna(method="bfill", inplace=True)
    #q_df.fillna(method="pad", inplace=True)

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


def getSetupDataLocalDB(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)

    if sess > 9999:
        p = os.sep.join([p, 'NES_RAW/NES/'])

    radar_file = radar_cpx_file(sess)
    cpx, setup_fs = load_radar_data(radar_file)

    phase = compute_phase(cpx)
    #np.save('/Neteera/Work/homes/dana.shavit/phase.npy', phase, allow_pickle=True)

    bins = None
    phase_df = pd.DataFrame(phase)
    i = cpx.real
    q = cpx.imag
    i_df = pd.DataFrame(i)
    q_df = pd.DataFrame(q)

    bins = load_radar_bins(radar_file)

    phase_df.interpolate(method="linear", inplace=True)
    phase_df.fillna(method="bfill", inplace=True)
    phase_df.fillna(method="pad", inplace=True)

    #i_df.interpolate(method="linear", inplace=True)
    #i_df.fillna(method="bfill", inplace=True)
    #i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    #q_df.fillna(method="bfill", inplace=True)
    #q_df.fillna(method="pad", inplace=True)

    ph = phase_df.to_numpy()
    i_np = i_df.to_numpy()
    q_np = q_df.to_numpy()
    respiration = compute_respiration(ph.flatten())
    #UP = 1
    #DOWN = 50
    #respiration = sp.resample_poly(respiration, UP, DOWN)

    #I = sp.resample_poly(i_np, UP, DOWN)
    #Q = sp.resample_poly(q_np, UP, DOWN)

    setup_fs = 500#int((500 * UP) / DOWN)
    return respiration, setup_fs, bins, i_np, q_np


def getSetupRespirationLocalDBDebugWithTimestamp(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)
    raw_dir = os.sep.join([p, 'NES_RAW'])

    radar_file = radar_cpx_file(sess)
    cpx, setup_fs = load_radar_data(radar_file)

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

    #i_df.interpolate(method="linear", inplace=True)
    #i_df.fillna(method="bfill", inplace=True)
    #i_df.fillna(method="pad", inplace=True)
    #q_df.interpolate(method="linear", inplace=True)
    #q_df.fillna(method="bfill", inplace=True)
    #q_df.fillna(method="pad", inplace=True)

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

def getApneaSegments(setup:int, sig_len: int, fs_new: float):
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
    print(np.unique(apnea_ref))
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

    apneas_mask = np.zeros(sig_len)

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

def computeReferenceAHI(respiration, apnea_segments, fs):
    n_hours = int(np.ceil(len(respiration)/(fs*60*60)))
    hours = [60 * 60 * fs * i for i in range(0, n_hours+1)]
    ahi_gt = np.zeros(n_hours)
    for i in range(n_hours):
        for a in apnea_segments:
            if a[1] > hours[i] and a[1] < hours[i+1]:
                ahi_gt[i] += 1
    return ahi_gt


def count_apneas_in_chunk(start_t, end_t, apnea_segments):
    count = 0
    if not apnea_segments:
        return 0
    for a in apnea_segments:

        if a[0] in range(start_t, end_t) and a[1] in range(start_t, end_t):
            count += 1
    #print(hypopneas, count)
    return count

def count_events_in_chunk(start_t, end_t, apnea_segments, event_list=None):
    count = 0

    for a in apnea_segments:
        if a[0] in range(start_t, end_t) and a[1] in range(start_t, end_t):
            if event_list:
                if a[3] in event_list:
                    count += 1
            else:
                count +=1


    return count


def create_AHI_training_data_no_wake(respiration, apnea_segments, wake_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []

    for i in range(time_chunk, len(respiration), step):
        wake_perc = 1.0 - (np.sum(wake_ref[int((i - time_chunk)/fs):int(i/fs)])/(time_chunk/fs))
        v = 1;
        if wake_perc > 0.75:
            v = v * 0
        else:
            v = v * 1

        seg = respiration[i - time_chunk:i]
        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        y.append(num_apneas)
        valid.append(v)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        valid = np.stack(valid)
        print(X.shape, y.shape)
        return X, y, valid

    return X, y, valid


def create_AHI_training_data_no_wake_no_empty(respiration, apnea_segments, wake_ref, empty_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []
    print("empty ref None", empty_ref is None)

    for i in range(time_chunk, len(respiration), step):
        wake_perc = 1.0 - (np.sum(wake_ref[int((i - time_chunk)/fs):int(i/fs)])/(time_chunk/fs))

        v = 1;
        if wake_perc > 0.75:
            #print(i, wake_perc)
            v = 0

        start_fs = int((i - time_chunk) / fs)
        end_fs = int(i / fs)
        if empty_ref is not None and end_fs < len(empty_ref):
            empty_ref = empty_ref[start_fs:end_fs]
            print(np.sum(empty_ref) / len(empty_ref))
            if np.sum(empty_ref) / len(empty_ref) > 0.3:
                print(np.sum(empty_ref) / len(empty_ref))
                v = 0
        seg = respiration[i - time_chunk:i]
        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        if len(seg) != time_chunk:
            continue


        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        y.append(num_apneas)
        valid.append(v)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)

        valid = np.stack(valid)
        print(X.shape, y.shape, np.sum(valid)/len(valid))
        return X, y, valid
    return X, y, valid


def create_AHI_training_data(respiration, apnea_segments, time_chunk, step, scale):
    X = []
    y = []
    valid = []
    #sleep_start = max(time_chunk, sleep_start) if sleep_start else time_chunk
    #sleep_end = min(len(respiration), sleep_end) if sleep_end else len(respiration)

    for i in range(time_chunk, len(respiration), step):

        seg = respiration[i - time_chunk:i]
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        y.append(num_apneas)
        valid.append(1)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        valid = np.stack(valid)
        return X, y, valid

    return X,y, valid
#

def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 parser.add_argument('-save_path', metavar='Location', type=str, required=True, help='location of saved data')
 parser.add_argument('--scale', action='store_true', help='Scale data to m=0 s=1')
 return parser.parse_args()

# apnea_class = {'missing': -1,
#         'Normal Breathing': 0,
#         'normal': 0,
#         'Central Apnea': 1,
#         'Hypopnea': 2,
#         'Mixed Apnea': 3,
#         'Obstructive Apnea': 4,
#         'Noise': 5}


if __name__ == '__main__':

    args = get_args()

    if args.scale:
        save_path = os.path.join(args.save_path, 'scaled')
    else:
        save_path = os.path.join(args.save_path, 'unscaled')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    output_setups = []
    nwh_company = db.setup_by_project(db_api.Project.nwh)
    bed_top = db.setup_by_mount(db_api.Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=db_api.Sensor.nes, value=db_api.Validation.invalid)
    setups = db.setup_nwh_benchmark()#set(bed_top) & set(nwh_company) - set(invalid) - {6283}
    for s in setups:
        natus_validity = db.setup_data_validity(s, db_api.Sensor.natus)
        if natus_validity in ['Confirmed', 'Valid']:
            output_setups.append(s)
    sessions = db.setup_nwh_benchmark()
    print(sessions)

    for i_sess, sess in enumerate(sessions):
        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(sessions)), "::::::::")
        # if sess in [8920, 8757, 8719, 9100, 6284, 9003, 9188, 8998, 9093, 8580, 8995, 8674, 8705]:
        #     continue
        respiration, fs_new = getSetupRespiration(sess)

        apnea_segments = getApneaSegments(sess, respiration, fs_new)
        X = []
        y = []
        chunk_size_in_minutes = 15
        time_chunk = fs_new * chunk_size_in_minutes * 60
        step = fs_new * 5 * 60

        X, y, _ = create_AHI_training_data(respiration, apnea_segments, time_chunk, step, args.scale)
        # for i in range(time_chunk, len(respiration), step):
        #     X.append(respiration[i - time_chunk:i])
        #     y.append(count_apneas_in_chunk(start_t= i - time_chunk,end_t=i, apnea_segments=apnea_segments))
        #     print(sess, i, y[-1])
        # session_id = []
        #
        # try:
        #     print(len(X))
        #     X = np.stack(X)
        #     y = np.stack(y)
        # except:
        #     print(sess, "failed to stack ")
        #     continue

        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) +  '_y.npy'), y, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)

        print("saved training data")
