from Tests.Utils.LoadingAPI import load_nes, load_reference, load_phase
from Tests.vsms_db_api import *
import scipy.signal as sp
import glob
from typing import Tuple
#from Tests.NN.create_apnea_count_AHI_data import compute_respiration
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sp
from scipy import linalg, signal
apnea_class = {'missing': -1,
                   'Normal Breathing': 0,
                   'normal': 0,
                   'Central Apnea': 1,
                   'Central': 1,
                   'Hypopnea': 2,
                   'Obstructive Hypopnea': 2,
                   'Mixed Apnea': 3,
                   'Obstructive Apnea': 4,
                   'Noise': 5,
                   '':-1,
                   'Apnea': 4}

def radar_phase_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=Sensor.nes)
    dir_radar = os.path.dirname(tlog_files_radar[0])
    print(glob.glob(os.path.join(dir_radar, "*.npy")))

    phase_file = [f for f in glob.glob(os.path.join(dir_radar, "*.npy")) if 'phase' in f][0]
    print("PHASE FN", phase_file)
    return phase_file


def qnormalize(x, lo=0.1, hi=0.9):
    """Squeeze the signal x to the range (-1,1) using quantiles."""
    qlo, qhi = np.quantile(x, lo), np.quantile(x, hi)
    return 2 * (x - qlo) / (qhi - qlo) - 1



def compute_respiration(phase: np.ndarray) -> np.ndarray:
    """Filter the phase to get the respiration signal."""
    sos = signal.butter(
        2, [0.05, 3.33], btype="bp", output="sos", fs=500.0
    )

    resp = signal.sosfiltfilt(sos, phase)
    return qnormalize(resp)

def load_radar_data(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    x = X[:, np.argmax(np.var(X, axis=0))]
    return x, 1 / Ts

def getSetupRespiration(sess: int):
    """return respiration displacement signal for setup"""
    phase_fn = radar_phase_file(sess)
    #print(phase_fn)
    phase = np.load(phase_fn, allow_pickle=True)
    #print(sess, type(phase))
    if isinstance(phase, pd.core.series.Series):
        phase = phase.to_numpy()

    respiration = compute_respiration(phase)
    UP = 1
    DOWN = 50
    respiration = sp.resample_poly(respiration, UP, DOWN)
    setup_fs = int((500 * UP) / DOWN)
    return respiration, setup_fs


def getWakeSegments(ss_ref):
    """Compute wake segments per setup, return segments and binary wake/sleep mask"""

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


def getApneaSegments(apnea_ref, respiration, fs_new):

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

        apnea_segments.append([start_idx, end_idx, apnea_duration[a_idx], apnea_type[a_idx]])
    return apnea_segments

def load_data_from_db(idx, db):
    print(f'loading data from db setup {idx}')
    data = load_nes(idx, db)
    if len(data) == 4:
        return data
    db_dist = db.setup_distance(idx)
    if db_dist:
        data['dist'] = db_dist
    data['gap'] = 1 / data['framerate']
    return data

db =DB()

sess = 109867
db.update_mysql_db(sess)

setups = db.all_setups()
setups =[s for s in setups if 'NC' in db.setup_subject(s)]
MB_HQ = [109816, 109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897,
         109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966]
for sess in MB_HQ:

    db.update_mysql_db(sess)
    print(sess, db.mysql_db)
    data = load_nes(sess, db)


    respiration, fs_new = getSetupRespiration(sess)
    print(sess, "len(respiration)", len(respiration))

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    ax[0].plot(respiration)
    #there is only one reference per session. you have to look it up in the correct dir
    apnea_ref = load_reference(sess,'apnea', db, use_saved_npy=True)
    if apnea_ref is None:
        print("ref not ok")

    else:
        print(sess, "apnea_segments", len(apnea_ref))
        apnea_c = np.zeros(len(apnea_ref))
        for i in range(len(apnea_ref)):
            apnea_c[i] = apnea_class[apnea_ref[i]]
        apnea_segments = getApneaSegments(apnea_ref, respiration, 10)
        for s in apnea_segments:
            print(s[0]/10, s[1]/10, (s[1]-s[0])/10)

            c = 'red' if s[3] != 2.0 else 'green'
            ax[0].axvspan(s[0], s[1], color=c, alpha=0.3)
            # ax[1].axvspan(s[0], s[1], color=c, alpha=0.3)
            plt.title(str(sess))
    #wake_segments = getWakeSegments(sess)
    db.update_mysql_db(sess)
    # reference = load_reference(sess, 'chest', db)
    #
    # chest = reference.to_numpy()
    # UP = 25
    # DOWN = 256
    # chest[chest == None] = 0
    # chest = sp.resample_poly(chest, UP, DOWN)
    # plt.plot(chest)





    # for s in wake_segments[0]:
    #     c = 'yellow'
    #     ax[0].axvspan(10 * s[0], 10 * s[1], color=c, alpha=0.3)
    #     #ax[1].axvspan(fs_new * s[0], fs_new * s[1], color=c, alpha=0.3)
    plt.title(str(sess))
    plt.show()