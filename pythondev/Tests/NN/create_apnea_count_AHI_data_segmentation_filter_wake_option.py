# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
from sklearn import preprocessing
import os
from Tests.NN.create_apnea_count_AHI_data_segmentation import create_AHI_segmentation_training_data,  radar_cpx_file, load_radar_data, compute_phase,  compute_respiration, getSetupRespiration
import numpy as np
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests import vsms_db_api as db_api


db = db_api.DB()

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


def create_AHI_segmentation_training_data_no_wake(respiration, apnea_ref, wake_ref, time_chunk, step, scale, fs):
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

            #continue
        seg = respiration[i - time_chunk:i]
        len_labels = int(time_chunk / fs)
        labels = apnea_ref[int((i - time_chunk) / fs):int(i / fs)]
        if len(labels[labels < 0]):
            v = v * 0
        else:
            v = v * 1
            #continue
        if len(labels) != len_labels:
            continue

        if len(seg) != time_chunk:
            continue
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        y.append(labels)
        valid.append(v)
    if len(X):
        X = np.stack(X)
        y = np.stack(y)
        valid = np.stack(valid)
        print(X.shape, y.shape)
        return X, y, valid

    return X, y, valid


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
    nwh_company = db.setup_by_project(db_api.Project.nwh)
    bed_top = db.setup_by_mount(db_api.Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=db_api.Sensor.nes, value=db_api.Validation.invalid)
    setups =  set(nwh_company) - set(invalid) - {8705, 6283}
    #setups = set(bed_top) & set(nwh_company) - set(invalid) - {6283, 9096, 8734}
    for s in setups:
        natus_validity = db.setup_data_validity(s, db_api.Sensor.natus)
        if natus_validity in ['Confirmed', 'Valid']:
            sessions.append(s)
    bm_sessions = db.setup_nwh_benchmark()

    for i_sess, sess in enumerate(sessions):
        print(":::::::: processing session", sess, str(i_sess)+'/'+str(len(sessions)), "::::::::")
        if args.overwrite and os.path.isfile(os.path.join(save_path,str(sess) + '_X.npy')):
            print(sess, "done, skipping")
            continue
        respiration, fs_new = getSetupRespiration(sess)

        X = []
        y = []
        chunk_size_in_minutes = args.chunk
        time_chunk = fs_new * chunk_size_in_minutes * 60
        step = fs_new * args.step * 60

        ss_ref_class = get_sleep_stage_labels(sess)
        if len(ss_ref_class) == 0:
            print('ref bad')
            continue

        apnea_ref_class = get_apnea_labels(sess)
        if len(apnea_ref_class) == 0:
            print('ref bad')
            continue

        if args.filter_wake_from_train:

            X, y, valid = create_AHI_segmentation_training_data_no_wake(respiration=respiration, apnea_ref=apnea_ref_class, wake_ref=ss_ref_class, time_chunk=time_chunk, step=step, scale=args.scale, fs=fs_new)
        else:
            X, y, valid = create_AHI_segmentation_training_data(respiration=respiration, apnea_ref=apnea_ref_class,
                                                time_chunk=time_chunk, step=step,
                                                scale=args.scale, fs=fs_new)
        print(X.shape, y.shape)
        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y[valid == 1], allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X[valid == 1], allow_pickle=True)

        print("saved training data")
