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

from Tests.NN.create_apnea_count_AHI_data import delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
import glob
import scipy.signal as sp

db = DB()
home = '/Neteera/Work/homes/dana.shavit/Research/analysis/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707/acc_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_amp_with_filter7_selected_sessions/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_TrinityLT/save_zc_10sec_ec_win_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/vsm_nwh_1707_NWH/amp7_zc_10sec_selected_sessions/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_min_time_for_output_10/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_min_time_for_output_10_3/'

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/herzog_min_time_for_output_30/'

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_peaks_double/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/new_codes/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/low_hr_60/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/herzog_1206/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/low_hr_60_peaks_6_real_data/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/ec_thresh_200/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/ec_thresh_200/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1406_ec_th_250/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_apnea/'
fn = base_path[base_path[:base_path.rfind('/')].rfind('/')+1:-1] + '.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}
setups = [ 110761, 110762, 110763, 110764, 110771, 110772, 110773, 110774, 110775, 110776, 110777, 110777, 110778, 110779, 110780, 110781, 110782, 110783, 110784, 110785, 110786, 110787, 110788, 110789, 110790, 110791, 110792, 110793, 110794, 110795, 110796, 110797, 110798, 110799, 110800, 110801, 110802, 110803, 110804, 110805, 110806, 110807]
print(setups)
setups = [110761, 110762, 110763, 110764]
setups = [ 110994, 110995, 110996, 110997, 110998, 110999, 111000,   111001]

setups = [111002,111003,111004,111005,111006,111007,111008,111009]
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_apnea/'
setups = [110851, 110939, 110940, 110941, 110942, 110796, 110795, 110783, 110782, 110871, 110771]



base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_511/'
setups = [111020]
#
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_314_307_2/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_apnea/'
setups = [110939]
#
# setups = [110994,110995,110996,110997,110998,110999,111000,111001,111002,111003,111004,111005,111006,111007,111009]
# # setups = [111000,111001,111002,111003]
# # setups = [110939]
# # setups = [111003]
# base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_1506_apnea/'
# setups = [110796]
# base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_ec_thresh_50/'
# setups=[ 110851, 110939 ,110940 ,110941 ,110942, 110796, 110795, 110783, 110782 ,110871, 110771 ,111002 ,111003 ,111004 ,111005 ,111006 ,111007,  111009, 111020 ,111021,111008,]
#
# base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/szmc_ec_thresh_250_ecwin_10/'
#
# setups =[109761, 109762, 109763, 109764, 110851 ,110939, 110940, 110941, 110942, 110796, 110795, 110783, 110782, 110871, 110771, 111002, 111003, 111004 ,111005, 111006, 111007, 111008, 111009, 111020, 111021]
#
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/szmc_ec_thresh_250_SMI/'
setups=[110761,110762,110763,110764,110851,110939,110940,110941,110942,110796,110795,110783,110782,110871,110771,111002,111003,111004,111005,111006,111007,111008,111020]
setups=[110762,110763,110764]
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
 parser.add_argument('-down', metavar='window', type=int, required=False, help='downsample')
 parser.add_argument('-lp', metavar='window', type=float, required=False, help='lowpass')
 parser.add_argument('-hp', metavar='window', type=float, required=False, help='highpass')

 return parser.parse_args()


def getSegments(data_frame, seg_val, decimate=False):
    #
    countr = 0
    segments = []
    for k, v in data_frame[data_frame['stat'] == seg_val].groupby((data_frame['stat'] != seg_val).cumsum()):
        if decimate:
            segments.append([int(np.floor(v.index[0] / 2)), int(np.floor(v.index[-1] / 2))])
        else:
            segments.append([int(v.index[0]), int(v.index[-1])])
        countr += 1
    return segments


if __name__ == '__main__':
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    print(setups)
    #setups = [109870]
    d = []

    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(5, sharex=True, figsize = (14, 7))
        setup_row = []

        setup_row.append(sess)
        setup_row.append(db.setup_duration(sess))
        setup_row.append(db.setup_sn(sess)[0])

        ts = None
        use_ts = False

        sig, fs_new, bins, I, Q = getSetupRespirationCloudDBDebug(sess)
        nsec = int(np.floor(len(I)/fs_new))
        new_len = int(nsec*fs_new)
        I = I[:new_len]
        sig_seconds = I.reshape(nsec, fs_new)

        firstsec = 0
        lastsec = sig_seconds.shape[0]
        dupes = []
        for i in range(firstsec, lastsec):
            if i % 100 == 0:
                print(i, "/", lastsec)
            pattern1 = sig_seconds[i, 0:5]
            p1 = np.tile(pattern1, [nsec,1])

            diff = np.sum(np.abs(sig_seconds[:,0:5]-p1), axis=1)

            zero_idxs =  np.where(diff < 0.0001)
            if len(zero_idxs)>1:
                dupes.append([i, zero_idxs])
        print(dupes)


        firstsec = 0
        lastsec= sig_seconds.shape[0]
        dupes = []
        for i in range(firstsec, lastsec):
            if i % 100 == 0:
                print(i,"/",lastsec)
            pattern1 = sig_seconds[i,0:5]

            for j in range(i+1, lastsec, 1):
                pattern2 = sig_seconds[j,0:5]
                sc = np.sum(np.abs(pattern1-pattern2))
                if sc < 0.0001:
                    dupes.append([i,j,sc])
                    print(i,j,sc)
        print(dupes)
        print("done")