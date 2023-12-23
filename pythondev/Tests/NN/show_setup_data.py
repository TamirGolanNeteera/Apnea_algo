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
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/ec_save_zc_stats4_ec_benchmark/'
setups = db.setup_ec_benchmark()

fn = '_posture'+'.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}

#
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/dbg_lab_recordings_serial/'
setups = [11275]

#
# setups = [110761] # db.setup_nwh_benchmark()
# print(setups)
#
# setups = [11275, 11276, 11279, 11280]
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
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(4, sharex=False)

        ts = None
        use_ts = False

        if sess > 100000:
            #respiration, fs_new, bins, I, Q, = getSetupRespirationCloudDBDebug(sess)

            if use_ts:
                respiration, fs_new, bins, I, Q, ts = getSetupRespirationCloudDBDebugWithTimestamp(sess)
            else:
                respiration, fs_new, bins, I, Q = getSetupRespirationCloudDBDebug(sess)
            binss = np.repeat(bins, fs_new)

            if ts is not None:
                ax[2].plot(ts, binss, label='bin', linewidth=0.25)
            else:
                ax[2].plot(binss, label='bin', linewidth=0.25)
        else:
            try:
                respiration, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(sess)

            except:
                tlog_files_radar = db.setup_ref_path(setup=sess, sensor=Sensor.nes)
                dir_radar = os.path.dirname(tlog_files_radar[0])
                radar_file = glob.glob(os.path.join(dir_radar, "*.*"))[0]

                import pandas as pd
                df = pd.read_csv(
                    str(radar_file),
                    header=None,
                    skiprows=48,
                    skipfooter=13,
                    usecols=list(range(30)),
                    converters={0: lambda x: int(x[4:])},
                    engine='python'
                )
                I = df.iloc[:, 0::2].to_numpy()
                Q = df.iloc[:, 1::2].to_numpy()
                pd.DataFrame(I + 1j * Q)

                X = I + 1j * Q
                cpx = X[:, np.argmax(np.var(X, axis=0))]
                phase = compute_phase(cpx)

                bins = None
                phase_df = pd.DataFrame(phase)

                i_df = pd.DataFrame(I)
                q_df = pd.DataFrame(Q)

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

        session = db.session_from_setup(sess)

        ax[0].set_title(str(session) + " " + str(sess))#+' '+db.setup_subject(sess) + ' '+db.setup_sn(sess)[0])
#        apnea_segments = getApneaSegments(sess, respiration, fs_new)

        #fig, ax = plt.subplots(3, gridspec_kw={"wspace": 0.2, "hspace": 0.5})

        # ax[1].axvspan(tod[sess], tod[sess], color='red', alpha=1.0,
        #               label='TOD')

        pcol = ['yellow', 'orange', 'red', 'maroon', 'magenta']

        if ts is not None:
            ax[0].plot(ts,  respiration, label='displacement', linewidth=0.25)
            ax[1].plot(ts, I, label='I', linewidth=0.25)
            ax[1].plot(ts, Q, label='Q', linewidth=0.25)
        else:
            ax[0].plot(respiration, label='displacement', linewidth=0.25)
            ax[1].plot(I, label='I', linewidth=0.25)
            ax[1].plot(Q, label='Q', linewidth=0.25)

        if sess > 10000 and sess < 200575:
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
                df = pd.read_csv(os.sep.join([res_dir, vs]))
                rr = np.repeat(df['rr'].to_numpy(), fs_new)
                rr_hq = np.repeat(df['rr_hq'].to_numpy(), fs_new)
                hr = np.repeat(df['hr'].to_numpy(), fs_new)
                hr_hq = np.repeat(df['hr_hq'].to_numpy(), fs_new)
                status = df['stat']

                #ax[1].plot(ts, hr, label='hr')

                #ax[2].axvspan(75000,96000, color='cyan' , alpha=0.3, label='RR drop')
                ax[3].plot(hr_hq)
                ax[3].plot(rr_hq)
                empty = df['stat'] == 'Empty'
                running =df['stat'] == 'Running'
                lowrr = df['stat'] == 'Low-respiration'
                motion = df['stat'] == 'Motion'

                empty[empty == True] = 1
                empty[empty == False] = 0
                print("Percentage EC", sum(empty)/len(empty))

                empty = empty.to_numpy()
                es = getSegments(empty)

                for e in es:
                    ax[0].axvspan(e[0]*fs_new, e[1]*fs_new, color='red', alpha=0.3)
                    ax[1].axvspan(e[0]*fs_new, e[1]*fs_new, color='red', alpha=0.3)

                print("...")

                lowrr[lowrr == True] = 1
                lowrr[lowrr == False] = 0
                lowrr = lowrr.to_numpy()
                #es = getSegments(lowrr)
                # for e in es:
                #    ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.3)
                print("...")

                motion[motion == True] = 1
                motion[motion == False] = 0
                motion = motion.to_numpy()
                es = getSegments(motion)
                for e in es:
                    ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='blue', alpha=0.3)
                print("...")
            except:
                empty = None
        else:
            empty=None
            cols = {'empty chair': 'red', 'Empty': 'red', 'low_signal': 'yellow', 'Low_signal': 'yellow',
                    'motion': 'blue', 'running': 'green', 'Motion': 'blue', 'Running': 'green',
                    'unknown': 'grey', 'Unknown': 'grey', 'warming-up': 'olivedrab', 'Warming-up': 'olivedrab',
                    'missing': 'black', 'Low-respiration': 'black'}
            to_draw = {'Low-respiration': False, 'empty chair': True, 'Empty': True, 'low_signal': True,
                       'Low_signal': True, 'motion': True, 'running': True, 'Running': True, 'unknown': False,
                       'Unknown': False, 'Motion': True, 'warming-up': False, 'Warming-up': False, 'missing': False}

            comp_stat = None

            try:
                comp_stat = np.array(np.load(os.path.join(base_path+'dynamic', str(sess)+'_stat.data'), allow_pickle=True))
                for uu in np.unique(comp_stat):
                    s = pd.DataFrame(comp_stat, columns=['stat'])

                    vv = comp_stat == uu
                    stat_vec = np.zeros_like(vv)
                    stat_vec[vv == True] = 1

                    print("Percentage", uu, 100 * (sum(vv) / len(vv)))
                    print("comp_stat getting segments for", uu)
                    es = getSegments(s, uu)

                    if 'mpty' in uu:
                        empty_offline = len(es)

                    for e in es:
                        if to_draw[uu]:
                            ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)
                            # ax[1].axvspan(e[0] * fs_new, e[1] * fs_new, color=cols[uu], alpha=0.3)
                    print("done drawing comp_stat")
            except:
                print('no stat data for setup', sess)

            try:
                zc_min = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc_min.npy'), allow_pickle=True))
                zc_max = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc_max.npy'), allow_pickle=True))
                zc_mean = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc_mean.npy'), allow_pickle=True))
                zc_std = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc_std.npy'), allow_pickle=True))

                ax[2].plot(zc_min, label='zc_min')
                ax[2].plot(zc_max, label='zc_max')
                ax[2].plot(zc_mean, label='zc_mean')
                ax[2].plot(zc_std, label='zc_std')

            except:
                print('no zc_stats data for setup', sess)

            # try:
            #     rr_r = np.array(np.load(os.path.join(base_path+'dynamic', str(sess)+'_rr_reliability.npy'), allow_pickle=True))
            #     rr_r = np.repeat(rr_r, fs_new)
            #     rr_hq = np.array(np.load(os.path.join(base_path+'dynamic', str(sess)+'_rr_high_quality_indicator.npy'), allow_pickle=True))
            #     rr_hq = np.repeat(rr_hq, fs_new)
            #     ax[1].plot(rr_r, label='reliability')
            #     ax[1].plot(rr_hq, label='rr hq')
            # except:
            #     print('no rr hq data for setup', sess)


            # reason=None
            # try:
            #     reason = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_reject_reason.npy'), allow_pickle=True))
            #     reason = np.repeat(reason, fs_new)
            #
            #     ax[3].plot(reason, color='magenta', label='reason')
            #     ax[3].set_yticks(range(-1,5,1))
            # except:
            #     print('no reject data for setup', sess)
            # zc=None
            # try:
            #     zc = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc_percentile.npy'), allow_pickle=True))
            #     zc = np.repeat(zc, fs_new)
            #     ax[1].plot(zc, color='orange', label='zc percentile')
            #
            #     #ax[3].set_yticks(range(-1,5,1))
            # except:
            #     print('no zc data for setup', sess)
            # try:
            #     #iamp = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_sig_diff_amp.npy'), allow_pickle=True))
            #     #iamp = np.repeat(iamp, fs_new)
            #     qamp = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_sig_amp.npy'), allow_pickle=True))
            #     qamp = np.repeat(qamp, fs_new)
            #
            #     #ax[3].plot(iamp, label='sig diff amp')
            #     ax[3].plot(qamp, label='sig amp')
            #
            #     #ax[3].set_yticks(range(-1,5,1))
            # except:
            #     print('no qual data for setup', sess)



            if empty is not None:
                #print(empty)
                #empty_array = np.zeros(len(empty))
                #print(len(empty[empty == 'empty chair']))
                #empty_array[empty == 'empty chair'] = 1
                es = getSegments(empty)
                print("Percentage EC", sum(empty) / len(empty))
                for e in es:
                    ax[0].axvspan(e[0]*fs_new, e[1]*fs_new, color='magenta', alpha=0.6)
                    #ax[2].axvspan(e[0]*fs_new, e[1]*fs_new, color='red', alpha=0.2)
                    #print(e[1]-e[0])
        ts_str = '_'
        if ts is not None:
            ts_str = '_ts_'

        ax[1].legend()
        ax[1].axhline(y=1000, color='red')
        ax[1].axhline(y=-1000, color='red')
        #ax[1].legend()
        #ax[3].legend()
        ax[0].legend()
        session = db.session_from_setup(sess)
        #plt.savefig(home+str(session)+"_"+str(sess)+fn, dpi=1000)
        #print("saved", home+str(sess)+fn)
        plt.show()
        plt.close()

