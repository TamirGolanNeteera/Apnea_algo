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

from Tests.NN.create_apnea_count_AHI_data import delays, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebugWithTimestamp, getSetupRespirationLocalDBDebugWithTimestamp#,getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, getApneaSegments
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
db = DB()


def create_AHI_regression_training_data_from_annotation(respiration, apnea_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []
    apnea_ref['start_t'] = fs*apnea_ref['onset']
    apnea_ref['end_t'] = apnea_ref['start_t'] + apnea_ref['duration']*fs
    for i in range(time_chunk, len(respiration), step):

        v = 1
        seg = respiration[i - time_chunk:i]
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
            #continue
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        #print(np.mean(seg), np.median(seg), np.std(seg))
        #print(apneas_df.loc[(apneas_df['start_t'] >= (i - time_chunk)) & (apneas_df['end_t'] <i)])
        num_apneas = len(apnea_ref.loc[(apnea_ref['start_t'] >= (i - time_chunk)) & (apnea_ref['end_t'] <i)])
        #num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        y.append(num_apneas)

        # plt.plot(preprocessing.robust_scale(seg))
        # plt.title(str(num_apneas))
        # plt.show()

        valid.append(v)
    if len(X):
        print(y)
        X = np.stack(X)
        y = np.stack(y)

        valid = np.stack(valid)
        #print(np.count_nonzero(valid))
        return X, y, valid

    return X,y, valid


def create_AHI_regression_training_data_MB_phase(respiration, apnea_ref, empty_seconds, time_chunk, step, scale, fs):
    print("in")
    X = []
    y = []
    valid = []
    apnea_segments = []

    #empty_seconds = [max(empty_seconds[i], empty_bins[i]) for i in range(len(empty_seconds))]

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()

    apnea_diff = np.diff(apnea_ref, prepend=0)

    apnea_changes = np.where(apnea_diff)[0]
    apnea_duration = apnea_changes[1::2] - apnea_changes[::2]  # apneas[:, 1]
    apnea_idx = apnea_changes[::2]  # np.where(apnea_duration != 'missing')
    apnea_end_idx = apnea_changes[1::2]
    apnea_type = 1  # apneas[:, 2]

    for a_idx, start_idx in enumerate(apnea_idx):

        if float(apnea_duration[a_idx]) == 0.0:
            continue
        end_idx = apnea_end_idx[a_idx]

        apnea_segments.append([start_idx, end_idx, apnea_duration, 1    ])
    print(len(apnea_segments), "apneas in setup")


    for i in range(time_chunk, len(respiration), step):

        v = 1
        seg = respiration[i - time_chunk:i]
        start_fs = int((i - time_chunk)/fs)
        end_fs = int(i/fs)
        if empty_seconds is not None and end_fs < len(empty_seconds):
            empty_ref = empty_seconds[start_fs:end_fs]

            if np.sum(empty_ref)/len(empty_ref) > 0.3:
                v = 0
            #print(start_fs, end_fs, np.sum(empty_ref), len(empty_ref), v)
        if (seg == -100).any():
            v = 0
        if np.mean(seg) < 1e-4 and np.std(seg) < 1e-5:
            v = 0
        if len(seg) != time_chunk:
            continue


        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=start_fs, end_t=end_fs, apnea_segments=apnea_segments)

        y.append(num_apneas)
        valid.append(v)

    if len(X):
        X = np.stack(X)
        y = np.stack(y)

        valid = np.stack(valid)
        print(np.count_nonzero(valid))
        print(len(X), len(y), np.sum(valid) / len(valid))
        return X, y, valid


    return X,y, valid
#
#

def get_empty_seconds_mb(setup):
    empty = None
    try:
        db.update_mysql_db(setup)
        p = db.setup_dir(setup)
        ref_dir = os.sep.join([p, 'NES_RES'])
        print(ref_dir)
        csv_file = fnmatch.filter(os.listdir(ref_dir), '*_VS.csv')[0]
        df = pd.read_csv(os.sep.join([ref_dir, csv_file]))
        empty = np.zeros(len(df['stat']))
        empty[df['stat'] == 'Empty'] = 1
    except:
        print("something broken in loading res")
    return empty


def load_apnea_ref_from_annotations(setup, db):
    try:
        s = db.session_from_setup(setup)
        apnea = load_reference(setup, 'apnea', db)

        if setup in delays.keys():
            apnea = apnea[delays[setup]:]
        if apnea is not None:
            return apnea

        data_setup = min(db.setups_by_session(s))
        p = db.setup_dir(data_setup)
        ref_dir = os.sep.join([p, 'REFERENCE/RESPIRONICS_ALICE6'])
        #print(ref_dir)
        #print(os.listdir(ref_dir))
        apnea = None
        for file in os.listdir(ref_dir):
            #print(file)
            if 'pnea.npy' in file:
                #print(os.path.join(ref_dir, file))
                anno_path = os.path.join(ref_dir, file)
                #print(anno_path)
                apnea = np.load(anno_path, allow_pickle=True)
                print('loaded apnea, setup', setup)
                break
        if len(apnea) > 0:
            #print(apnea.keys())
            print('ok')
            return apnea
        else:
            print(setup, "not ok, no ref")
            return None
    except:
        print(setup, "not ok exception")
        return None

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



if __name__ == '__main__':

    #args = get_args()


    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']
    home = '/Neteera/Work/homes/dana.shavit/'

    setups = [ 110580, 110583, 110581]

    print(setups)
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(2, sharex=True)
        if sess > 100000:
            respiration1, fs_new, bins, I, Q, ts = getSetupRespirationCloudDBDebugWithTimestamp(sess)
            binss = np.repeat(bins, fs_new)
            #ax[0].plot(ts, binss, label='bin')
        else:
            respiration, fs_new, bins, I, Q, ts = getSetupRespirationLocalDBDebugWithTimestamp(sess)

        ax[0].set_title('Respiration '+str(sess)+' '+db.setup_subject(sess) + ' '+db.setup_sn(sess)[0])
        alerts = pd.read_csv('/Neteera/Work/homes/dana.shavit/FromDG/hertzog/'+str(sess)+'.csv')

        alerts['startTime'] = pd.to_datetime(alerts['startTime'])
        alerts['endTime'] = pd.to_datetime(alerts['endTime'])

        if sess == 110580:

            for i, a in enumerate(alerts['startTime']):
                if i == 0:
                    continue
                b = alerts['endTime'][i]
                if alerts['thresholdPreposition'][i] == 'BELOW':
                    color='magenta'
                else:
                    color='red'
                ax[0].axvspan(a, b, color=color, alpha=0.3, label=str(alerts['thresholdPreposition'][i])+' '+str(alerts['thresholdValue'][i]))

        tod = {110580: pd.to_datetime('2022-09-28 15:30:00'), 110581: pd.to_datetime('2022-09-08 03:41:00'), 110583: pd.to_datetime('2022-09-07 19:03:00')}
        #
        ax[0].axvspan(tod[sess], tod[sess], color='red', alpha=1.0,
                       label='TOD')
        #ax[0].plot(ts, respiration, label='displacement')
        #ax[0].plot(ts, I, label='I')
        #ax[0].plot(ts, Q, label='Q')
        if sess > 9200:
            p = db.setup_dir(sess)
            raw_dir = os.sep.join([p, 'NES_RAW'])
            res_dir = os.sep.join([p, 'NES_RES'])

            onlyfiles = listdir(res_dir)
            vs = [f for f in onlyfiles if 'VS' in f and 'csv' in f]

            if len(onlyfiles) ==1:
                res_dir = os.sep.join([res_dir, onlyfiles[0]])
                onlyfiles = listdir(res_dir)
                vs = [f for f in onlyfiles if 'results' in f and 'csv' in f]

            vs = vs[0]
            df = pd.read_csv(os.sep.join([res_dir, vs]))
            rr = np.repeat(df['rr'].to_numpy(), fs_new)
            rr_hq = df['rr_hq'].to_numpy()
            hr = np.repeat(df['hr'].to_numpy(), fs_new)
            hr_hq = df['hr_hq'].to_numpy()
            status = df['stat']

            ax[1].plot(ts, hr, label='hr')

            #ax[2].axvspan(75000,96000, color='cyan' , alpha=0.3, label='RR drop')
            hs = getSegments(hr_hq)
            #for e in hs:
            #    ax[1].axvspan(e[0]*fs_new, e[1]*fs_new, color='green', alpha=0.3)
            #ax[1].plot(hr_hq, label='hr_hq')

            ax[0].plot(ts, rr, label='rr')
            print("rr_hq", rr_hq[0:10])
            print("hr_hq", hr_hq[0:10])
            hs = getSegments(rr_hq)
            #for e in hs:
            #    ax[2].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.3)
            #ax[2].plot(rr_hq, label='rr_hq')

            ax[0].legend(fontsize=16, loc='upper left')
            ax[0].axhline(y=12, color='magenta', label='12 BPM')

            ax[1].legend(fontsize=16, loc='upper left')
            #ax[0].set_xticklabels(fontsize=20)
            #ax[1].set_xticklabels(fontsize=20)
            #ax[3].legend()
            #ax[0].legend()

            empty = df['stat'] == 'Empty'
            running =df['stat'] == 'Running'
            lowrr = df['stat'] == 'Low-respiration'
            motion = df['stat'] == 'Motion'

            #for d in df.iterrows():
                # if d[1]['stat'] == 'Empty':
                #     #ax[0].axvspan(d[0]*fs_new, (d[0]+1)*fs_new, color='red', alpha=0.3)
                # elif d[1]['stat'] == 'Motion':
                #     ax[0].axvspan(d[0]*fs_new, (d[0]+1)*fs_new, color='blue', alpha=0.3)
                #     #ax[3].axvspan(d[0]*fs_new, (d[0]+1)*fs_new, color='blue', alpha=0.3)
                # elif d[1]['stat'] == 'Low-respiration':
                #     ax[0].axvspan(d[0]*fs_new, (d[0]+1)*fs_new, color='green', alpha=0.3)
                #     #ax[2].axvspan(d[0]*fs_new, (d[0]+1)*fs_new, color='green', alpha=0.3)
            empty[empty == True] = 1
            empty[empty == False] = 0
            empty = empty.to_numpy()
            #es = getSegments(empty)
            #for e in es:
            #    ax[0].axvspan(e[0]*fs_new, e[1]*fs_new, color='red', alpha=0.3)
            print("...")

            lowrr[lowrr == True] = 1
            lowrr[lowrr == False] = 0
            lowrr = lowrr.to_numpy()
            #es = getSegments(lowrr)
            #for e in es:
            #    ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.3)
            print("...")

            motion[motion == True] = 1
            motion[motion == False] = 0
            motion = motion.to_numpy()
            #es = getSegments(motion)
            #for e in es:
            #    ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='blue', alpha=0.3)
            print("...")

        plt.savefig(home+str(sess)+'_'+db.setup_sn(sess)[0]+'.png', dpi=300)
        plt.show()
        plt.close()

        # for i in range(9000,len(I) ,9000):
        #     I_seg = I[i-9000:i]
        #     Q_seg = Q[i-9000:i]
        #     plt.figure()
        #     plt.scatter(I_seg,Q_seg, s=1)
        #     plt.title(str(sess)+'_'+str(i-9000)+":"+str(i))
        #     nf = os.path.join(home, str(sess)+'_'+str(i-9000)+"-"+str(i)+'__.png')
        #     plt.savefig(nf, dpi=300)
        #     plt.close()
        # continue
