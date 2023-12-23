# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch
from os import listdir
from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from pylibneteera.filters import Filter
from pylibneteera.float_indexed_array import TimeArray, FrequencyArray
from Tests.NN.create_apnea_count_AHI_data import radar_cpx_file, load_radar_data, delays, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
import glob
import scipy.signal as sp

db = DB()
home = '/Neteera/Work/homes/dana.shavit/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707/acc_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_amp_with_filter7_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_10sec_ec_win_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_10sec_selected_sessions/'

fn = '_posture'+'.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}



#setups = [11030, 11059, 11075, 11085, 11087]
def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 parser.add_argument('-save_path', metavar='Location', type=str, required=True, help='location of saved data')
 parser.add_argument('--scale', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--show', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')
 parser.add_argument('-down', metavar='window', type=int, required=False, help='downsample')
 parser.add_argument('-lp', metavar='window', type=float, required=False, help='lowpass')
 parser.add_argument('-hp', metavar='window', type=float, required=False, help='highpass')

 return parser.parse_args()



def preprocess(data, fs):
    """ Downsample, band pass in a high band, take abs and bandpass in the relevant band
    """
    data = TimeArray(data, 1/fs)
    fil_obj = Filter()
    high_band = [10,20]
    prep_params = {'to_freq': 100,   # Hz
                                     'high': 15,   # Hz
                                     'low': 35 / 60,  # Hz
                                     'loc_low_for_low_observation': 50 / 60,  # Hz
                                     'low_observation': 60,  # bpm
                                     'filter_order': 4,
                                     'num_taps_fir': 51,
                                     'cutoff_fir': 45,  # bpm
                                     'initial_sampling_rate_cut': 100,  # bpm
                                     'high_filter_pass_method': 'lfilter',
                                     'high_filter_pass_method_2nd': 'filtfilt',
                                     'lfilter_reduction_tail': 0.000,   # 15%
                                     # for standing this is 3 although it is filtfilt (optimized)
                                     'high_filter_degree': 3,
                                     'high_filter_degree_2nd': 1,

                                     }
    print(data.shape)
    x = fil_obj.fast_downsample_fir(data, int(data.fs), prep_params['initial_sampling_rate_cut'],
                                     prep_params['num_taps_fir'], prep_params['cutoff_fir'])
    fs = x.fs
    print(x.shape)
    x = fil_obj.butter_bandpass_filter(x, high_band[0], high_band[1], prep_params['high_filter_degree'],
                                       prep_params['high_filter_pass_method'], prep_params['lfilter_reduction_tail'])
    print("after filter 1", x.shape)
    x = np.abs(x)
    x = fil_obj.butter_bandpass_filter(x, prep_params['low'], prep_params['high'], prep_params['filter_order'])
    print("after filter 2", x.shape)
    return x


def getSetupHPPhase(sess: int):
    """return respiration displacement signal for setup"""
    db.update_mysql_db(sess)
    p = db.setup_dir(sess)

    if sess > 9999:
        p = os.sep.join([p, 'NES_RAW/NES/'])

    radar_file = radar_cpx_file(sess)
    cpx, setup_fs = load_radar_data(radar_file)

    phase = compute_phase(cpx)
    hp_phase = preprocess(phase, 500)
    return hp_phase



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


def getSegments(v):
    #

    segments = []

    v_diff = np.diff(v, prepend=0)

    v_changes = np.where(v_diff)[0]
    if len(v_changes) % 2:
        v_changes = np.append(v_changes, len(v))

    v_idx = v_changes[::2]  # np.where(apnea_duration != 'missing')
    v_end_idx = v_changes[1::2]


    for a_idx, start_idx in enumerate(v_idx):
        # if float(v_duration[a_idx]) == 0.0:
        #     continue
        end_idx = v_end_idx[a_idx]
        segments.append([start_idx, end_idx])
    return segments



def get_bins_nw(base_path, setup):
    bins_fn = str(setup)+'_estimated_range.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

def get_ec_reason(base_path, setup):
    bins_fn = str(setup)+'_reject_reason.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

def get_autocorr_data(base_path, setup):
    bins_fn = str(setup)+'acc_autocorr.npy'

    try:
        bins = np.load(os.sep.join([base_path, bins_fn]), allow_pickle=True)
    except:
        return None

    return bins

if __name__ == '__main__':
    #args = get_args()
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    setups = [11301,11302]#db.setup_nwh_benchmark()
    print(setups)

    print(setups)

    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(4, sharex=True)

        ts = None
        use_ts = False

        respiration, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(sess)
        delay = db.setup_delay(sess, sensor=Sensor.spo2)
        print("delay", delay)
        spo2 = load_reference(sess, 'spo2', db).fillna(method='bfill').to_numpy()

        spo2 = np.repeat(spo2, 100)
        fs_ref = 500
        ax[0].set_title(str(sess))#+' '+db.setup_subject(sess) + ' '+db.setup_sn(sess)[0])
        apnea_segments = getApneaSegments(sess, respiration, fs_new)

        pcol = ['yellow', 'orange', 'red', 'maroon', 'magenta']

        hp_phase = getSetupHPPhase(sess)
        if apnea_segments:
            for s in apnea_segments:
                c = 'blue'
                ax[0].axvspan(s[0], s[1], color=c, alpha=0.3)
                ax[1].axvspan(s[0], s[1], color=c, alpha=0.3)
                ax[2].axvspan(s[0], s[1], color='red', alpha=0.3)
        if ts is not None:
            ax[0].plot(ts,  np.repeat(respiration,10), label='displacement', linewidth=0.25)
            ax[1].plot(ts, np.repeat(I,10), label='I', linewidth=0.25)
            ax[1].plot(ts, np.repeat(Q,10), label='Q', linewidth=0.25)
        else:
            ax[0].plot(np.repeat(respiration,10), label='displacement', linewidth=0.25)
            ax[1].plot(np.repeat(I,10), label='I', linewidth=0.25)
            ax[1].plot(np.repeat(Q,10), label='Q', linewidth=0.25)


        #ax[1].legend()
        ax[2].plot(spo2, label='spo2', linewidth=0.5)
        #ax[0].legend()
        #ax[2].legend()
        ax[3].plot(hp_phase, label='hp_phase', linewidth=0.5)
        #plt.savefig(home+str(sess)+fn)
        print("saved", home+str(sess)+fn)
        plt.show()
        plt.close()

