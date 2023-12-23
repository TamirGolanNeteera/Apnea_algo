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
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/herzog_1206/'np.repeat
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/low_hr_60_peaks_6_real_data/'
#base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/ec_thresh_200/'
fn = base_path[base_path[:base_path.rfind('/')].rfind('/')+1:-1] + '.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}
setups = [ 110761, 110762, 110763, 110764, 110771, 110772, 110773, 110774, 110775, 110776, 110777, 110777, 110778, 110779, 110780, 110781, 110782, 110783, 110784, 110785, 110786, 110787, 110788, 110789, 110790, 110791, 110792, 110793, 110794, 110795, 110796, 110797, 110798, 110799, 110800, 110801, 110802, 110803, 110804, 110805, 110806, 110807]
#setups, = [110761, 110762, 110763, 110764, 110784, 110785, 110786, 110787, 110788, 110789, 110790, 110791, 110792, 110793, 110794, 110795, 110796, 110797, 110798, 110799, 110800, 110801, 110802, 110803, 110804, 110805, 110806, 110807]

#setups = [110761, 110762, 110763, 110764] # db.setup_nwh_benchmark()
print(setups)

#setups = [ 110781, 110782, 110783]
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

def getPostureSegments(setup:int, respiration: np.ndarray, fs_new: float):
    """compute posture segments per session"""
    posture_ref = None

    db.update_mysql_db(setup)
    if db.mysql_db  == 'neteera_cloud_mirror':
        posture_ref = load_reference(setup, 'posture', db)
    else:
        if setup in db.setup_nwh_benchmark():
            posture_ref = load_reference(setup, 'posture', db)
        else:
            session = db.session_from_setup(setup)
            setups = db.setups_by_session(session)
            for s in setups:
                if s in db.setup_nwh_benchmark():
                    posture_ref = load_reference(s, 'posture', db)

    #print(sess, type(posture_ref))
    if posture_ref is None:
        return None

    if isinstance(posture_ref, pd.core.series.Series):
        posture_ref = posture_ref.to_numpy()
    posture_from_displacement = np.zeros(len(posture_ref) * fs_new)

    posture_segments = []

    for i in range(len(posture_ref) - 1):
        if posture_ref[i] not in posture_class.keys():
            posture_from_displacement[i * fs_new:(i + 1) * fs_new] = -1
        else:
            posture_from_displacement[i * fs_new:(i + 1) * fs_new] = posture_class[posture_ref[i]]

    if posture_from_displacement[0] == -1:
        posture_diff = np.diff(posture_from_displacement, prepend=0)
    else:
        posture_diff = np.diff(posture_from_displacement, prepend=-1)

    posture_changes = np.where(posture_diff)[0]
    print(posture_changes)
#    posture_duration = posture_changes[1::2] - posture_changes[::2]  # postures[:, 1]
    posture_idx = posture_changes[::2]  # np.where(posture_duration != 'missing')
    posture_end_idx = posture_changes[1::2]
    posture_type = posture_from_displacement[posture_idx]  # postures[:, 2]

    postures_mask = np.zeros(len(respiration))

    for a_idx, start_idx in enumerate(posture_idx):
        try:
            if posture_type[a_idx] not in posture_class.values():
                print(posture_type[a_idx], "not in posture_class.values()")
                continue

            end_idx = posture_end_idx[a_idx]
            postures_mask[start_idx:end_idx] = 1

            posture_segments.append([start_idx, end_idx, end_idx-start_idx, posture_type[a_idx]])
        except:
            continue
    return posture_segments


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

    v_diff = np.diff(v, prepend=v[0])

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

        if sess > 100000:
            #respiration, fs_new, bins, I, Q, = getSetupRespirationCloudDBDebug(sess)

            if use_ts:
                respiration, fs_new, bins, I, Q, ts = getSetupRespirationCloudDBDebugWithTimestamp(sess)
            else:
                respiration, fs_new, bins, I, Q = getSetupRespirationCloudDBDebug(sess)
            binss = np.repeat(bins, fs_new)

            # if ts is not None:
            #     ax[2].plot(ts, binss, label='bin', linewidth=0.25)
            # else:
            #     ax[2].plot(binss, label='bin', linewidth=0.25)

        session = db.session_from_setup(sess)

        ax[0].set_title(str(session) + " " + str(sess) + ' '+db.setup_sn(sess)[0])
#
        pcol = ['yellow', 'orange', 'red', 'maroon', 'magenta']

        if ts is not None:
            ax[0].plot(ts,  respiration, label='displacement', linewidth=0.25)
            ax[1].plot(ts, I, label='I', linewidth=0.25)
            ax[1].plot(ts, Q, label='Q', linewidth=0.25)
        else:
            ax[0].plot(respiration, label='displacement', linewidth=0.25)
            ax[1].plot(I, label='I', linewidth=0.25)
            ax[1].plot(Q, label='Q', linewidth=0.25)


        p = db.setup_dir(sess)
        raw_dir = os.sep.join([p, 'NES_RAW'])
        res_dir = os.sep.join([p, 'NES_RES'])

        try:
            onlyfiles = listdir(res_dir)
            vs = [f for f in onlyfiles if 'VS' in f and 'csv' in f]

            if len(onlyfiles) ==1:
                res_dir = os.sep.join([res_dir, onlyfiles[0]])
                onlyfiles = listdir(res_dir)
                vs = [f for f in onlyfiles if 'results' in f and 'csv' in f]

            vs = vs[0]
            df = pd.read_csv(os.sep.join([res_dir, vs]))[::2]
            rr = np.repeat(df['rr'].to_numpy(), fs_new)
            rr_hq = df['rr_hq'].to_numpy()
            hr = np.repeat(df['hr'].to_numpy(), fs_new)
            hr_hq = df['hr_hq'].to_numpy()
            status = df['stat']
            hr_hq = hr_hq[::2]
            rr_hq = rr_hq[::2]
            #hr = hr[::2]
            #ax[1].plot(ts, hr, label='hr')

            #ax[2].axvspan(75000,96000, color='cyan' , alpha=0.3, label='RR drop')

            status = df['stat'].fillna('missing').to_numpy()
            hr_hq_rate_online = int(np.round(100 * len(hr_hq[hr_hq==1])/len(hr_hq[hr_hq>=0])))

            rr_hq_rate = int(np.round(100 * len(rr_hq[rr_hq==1])/len(rr_hq[rr_hq>=0])))
            hr_val_rate_online = int(np.round(100 * len(hr[hr>0])/len(hr)))
            print(hr_hq_rate_online, rr_hq_rate)
            ax[3].plot(hr, linewidth=0.5, alpha=0.5, color='blue', label='HR: HQ '+str(hr_hq_rate_online)+'%,  > -1 '+str(hr_val_rate_online)+'%')

            reason=None
            try:
                print("noise reason")
                reason = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_save_noise_reason.npy'), allow_pickle=True))
                reason = np.repeat(reason, fs_new)
                for u in np.unique(reason):
                    print(u, 100*len(reason[reason == u])/len(reason))
                ax[2].plot(reason, color='magenta', label='noise reason', linewidth=0.5)
                #ax[3].set_yticks(range(-1,5,1))
            except:
                print('no reject data for setup', sess)


            reason = None
            try:
                print("hr == -1 reason")
                reason = np.array(np.load(os.path.join(base_path + 'accumulated', str(sess) + '_quality_fail.npy'),
                                          allow_pickle=True))
                reason = np.repeat(reason, fs_new)
                for u in np.unique(reason):
                    print(u, 100*len(reason[reason == u])/len(reason))
                ax[4].plot(reason, color='magenta', label='LQ reason', linewidth=0.5)
                #ax[3].set_yticks(range(-1, 5, 1))
            except:
                print('no reject data for setup', sess)

            computed_stat = None
            try:
                comp_stat = np.array(np.load(os.path.join(base_path + 'dynamic', str(sess) + '_stat.data'),
                                          allow_pickle=True))
                cols = {'empty chair':'red', 'low_signal':'yellow', 'motion':'blue', 'running':'green', 'unknown':'grey', 'warming-up':'olivegreen'}
                to_draw = {'empty chair':True, 'low_signal':True, 'motion':True, 'running':True, 'unknown':False, 'warming-up':True}
                for uu in np.unique(comp_stat):
                    vv = comp_stat == uu
                    stat_vec = np.zeros_like(vv)
                    stat_vec[vv == True] = 1

                    print("Percentage",uu,  100 * (sum(vv) / len(vv)))
                    es = getSegments(stat_vec)
                    for e in es:
                        if to_draw[uu]:
                            ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)
                            ax[1].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)
                # empty_offline = int(np.round(100 * len(comp_stat[comp_stat=='empty chair']) / len(comp_stat)))
                # empty_online = int(np.round(100 * len(status[status=='empty chair']) / len(status)))
                # print(empty_online, empty_offline)
            except:
                print('no reject data for setup', sess)


            # cols = {'empty chair': 'red', 'low_signal': 'yellow', 'motion': 'blue', 'running': 'green',
            #         'unknown': 'grey', 'warming-up': 'olivegreen', 'warming-up': 'olivegreen' , 'missing': 'lightgray'}
            # to_draw = {'empty chair': True, 'low_signal': True, 'Low_signal': True, 'motion':True, 'running': True, 'Running': True, 'unknown': False,  'Unknown': False, 'Motion':True, 'warming-up': True, 'Warming-up': True, 'missing':True}
            #
            #
            #
            # for uu in np.unique(status):
            #     vv = status == uu
            #     stat_vec = np.zeros_like(vv)
            #     stat_vec[vv == True] = 1
            #
            #     print("Percentage", uu, 100 * (sum(vv) / len(vv)))
            #     es = getSegments(stat_vec)
            #     for e in es:
            #         if uu not in to_draw.keys():
            #             print(uu)
            #         if uu not in cols.keys():
            #             print(uu)
            #         if to_draw[uu]:
            #             ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)
            #             #ax[1].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)

            computed_hr = None
            try:
                comp_hr = np.array(np.load(os.path.join(base_path + 'dynamic', str(sess) + '_hr.npy'),
                                             allow_pickle=True))
                hr_hq = np.array(np.load(os.path.join(base_path + 'dynamic', str(sess) + '_hr_high_quality_indicator.npy'),
                                           allow_pickle=True))
                comp_hr = np.repeat(comp_hr, fs_new)
                hr_hq = np.repeat(hr_hq, fs_new)
                hr_hq_rate_offline = int(np.round(100 * len(hr_hq[hr_hq==1])/len(hr_hq[hr_hq>=0])))

                hr_val_rate_offline = int(np.round(100 * len(comp_hr[comp_hr>0])/len(comp_hr)))

                print(hr_hq_rate_offline, rr_hq_rate)

                setup_row.append(hr_hq_rate_online)
                setup_row.append(hr_hq_rate_offline)

                setup_row.append(hr_val_rate_online)
                setup_row.append(hr_val_rate_offline)

                ax[3].plot(comp_hr, linewidth=0.5, alpha=0.5, color='green',
                           label='HR: HQ ' + str(hr_hq_rate_offline) + '%,  > -1 ' + str(hr_val_rate_offline) + '%')

            except:
                print('no reject data for setup', sess)
        except:
            empty = None
        # setup_row.append(empty_online)
        # setup_row.append(empty_offline)

        d.append(setup_row)
        ts_str = '_'
        if ts is not None:
            ts_str = '_ts_'

        ax[1].legend()
        ax[1].axhline(y=1000, color='red', linewidth=0.5)
        ax[1].axhline(y=-1000, color='red', linewidth=0.5)
        ax[2].legend()
        ax[3].legend()
        ax[4].legend()
        ax[0].legend()
        session = db.session_from_setup(sess)
        outfolder = '/Neteera/Work/homes/dana.shavit/Research/analysis/low_hr_60_peaks_6/'
        plt.savefig(outfolder+str(session)+"_"+str(sess)+fn, dpi=300)

        #plt.show()
        plt.close()
    out_df = pd.DataFrame(d, columns=['Setup', 'Duration', 'SN', 'HR_HQ online', 'HR_HQ offline', 'Valid HR % online', 'Valid HR % offline'])#, 'empty % online', 'empty % fix'])
    out_df.to_csv('/Neteera/Work/homes/dana.shavit/Research/analysis/low_hr_60_peaks_6/'+fn+'.csv')
