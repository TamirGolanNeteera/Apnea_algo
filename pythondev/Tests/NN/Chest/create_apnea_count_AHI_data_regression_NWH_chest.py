# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse


from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from OpsUtils.Tests.Utils.LoadingAPI import load_reference
from OpsUtils.Tests.vsms_db_api import *
from OpsUtils.Tests.NN.create_apnea_count_AHI_data import count_apneas_in_chunk
from OpsUtils.pylibneteera.ExtractPhaseFromCPXAPI import get_fs

db = DB()

def get_sleep_stage_labels(setup:int):
    ss_ref = []
    if setup in db.setup_nwh_benchmark():
        ss_ref = load_reference(setup, 'sleep_stages', db)
    else:
        session = db.session_from_setup(setup)
        setups = db.setups_by_session(session)
        for s in setups:
            if s in db.setup_nwh_benchmark():
                ss_ref = load_reference(s, 'sleep_stages', db)

    #print(sess, type(ss_ref))
    if ss_ref is None:
        return []

    if isinstance(ss_ref, pd.core.series.Series):
        ss_ref = ss_ref.to_numpy()


    ss_ref_class = np.zeros_like(ss_ref)

    for i in range(len(ss_ref)):
        if ss_ref[i] == 'W':
            ss_ref_class[i] = 0
        elif ss_ref[i] == None:
            ss_ref_class[i] = 0
        else:
            ss_ref_class[i] = 1
    return ss_ref_class

def get_apnea_labels(setup:int):
    apnea_ref = []
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
        return []

    if isinstance(apnea_ref, pd.core.series.Series):
        apnea_ref = apnea_ref.to_numpy()


    apnea_ref_class = np.zeros_like(apnea_ref)

    for i in range(len(apnea_ref)):
        if apnea_ref[i] not in apnea_class.keys():
            apnea_ref_class[i] = -1
        else:
            apnea_ref_class[i] = apnea_class[apnea_ref[i]]
    return apnea_ref_class


def create_AHI_regression_training_data_MB_chest(respiration, apnea_ref, time_chunk, step, scale, fs):
    X = []
    y = []
    valid = []
    apnea_segments = []

    apnea_ref = apnea_ref.to_numpy()
    print(np.unique(apnea_ref))
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
    print(len(apnea_segments))
    for i in range(time_chunk, len(respiration), step):
        v = 1
        seg = respiration[i - time_chunk:i]
        if (seg == -100).any():
            v = 0
        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.robust_scale(seg))
        else:
            X.append(seg)

        num_apneas = count_apneas_in_chunk(start_t=i - time_chunk, end_t=i, apnea_segments=apnea_segments)

        y.append(num_apneas)
        valid.append(v)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        print(np.unique(y))
        valid = np.stack(valid)
        print(np.count_nonzero(valid))
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
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')
 parser.add_argument('--filter_wake_from_train', action='store_true',  required=False, help='filter_wake_from_train')
 parser.add_argument('-company',  type=str,  required=True, help='Hospital Origin data (MB or NWH)')

 return parser.parse_args()

apnea_class = {'missing': -1,
        'Normal Breathing': 0,
        'normal': 0,
        'Central Apnea': 1,
        'Hypopnea': 1,
        'Mixed Apnea': 1,
        'Obstructive Apnea': 1,
        'Noise': -1}


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

    if args.company == 'MB':
        db = DB('neteera_cloud_mirror')
        vs = 'abdominal'
        setups = db.all_setups()
    else:
        db = DB('neteera_db')
        setups = db.setup_by_company(args.company)
        vs = 'chest'

    sessions = set([db.session_from_setup(setup) for setup in setups])
    setups = [min(db.setups_by_session(session)) for session in sessions]

    for i_sess, sess in enumerate(setups):
        setups = db.setups_by_session(db.session_from_setup(sess))
        sn = [db.setup_sn(s) for s in setups]

        broken = False
        for s in setups:
            if s != sess and db.setup_sn(s) == db.setup_sn(sess):
                broken = True

        if broken:
            print(sess, "broken session")
            continue


        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(setups)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path,str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue

        try:
            reference = load_reference(sess, ['apnea', vs], db).fillna(method='ffill')
        except:
            print(sess, "not ok")
            continue
        reference = reference.asfreq('100ms', method='pad')
        fs_ref = get_fs(reference)

        apnea_with_gaps = reference.apnea


        apnea_ref = (apnea_with_gaps.notna() & (apnea_with_gaps != 'normal')) * 1
        #plt.plot(range(len(abs)), abs)#2*abs/max(abs))
        #plt.plot(range(len(apnea_with_gaps)), apnea_ref)



        #plt.title(f"setup id: {sess}")
        # create_dir(f'{SAVE_FOLDER}/{idx}/')
        # plt.savefig(f'{SAVE_FOLDER}/{idx}/{idx}_chest_apnea_with_gaps.png')
        #plt.show()
        #plt.close()
        #continue
        #respiration, fs_new = getSetupRespiration(sess)

        X = []
        y = []
        chunk_size_in_minutes = args.chunk
        time_chunk = fs_ref * chunk_size_in_minutes * 60
        step = fs_ref * args.step * 60

        X, y, valid = create_AHI_regression_training_data_MB_chest(respiration=reference[vs], apnea_ref=apnea_ref,
                                                time_chunk=time_chunk, step=step,
                                                scale=args.scale, fs=fs_ref)
        print(X.shape, y.shape)
        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        np.save(os.path.join(save_path, str(sess) + '_valid.npy'), valid, allow_pickle=True)
        print("saved training data")
