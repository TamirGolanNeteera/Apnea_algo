# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import scipy.signal as sp
from scipy import linalg, signal

from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import count_apneas_in_chunk
from pylibneteera.ExtractPhaseFromCPXAPI import get_fs
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


def create_AHI_regression_training_data_NW_chest(respiration, apnea_ref, time_chunk, step, scale, fs):
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
        num_apneas = len(apneas_df.loc[(apnea_ref['start_t'] >= (i - time_chunk)) & (apnea_ref['end_t'] <i)])
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

 return parser.parse_args()

apnea_class = {'missing': -1,
        'Normal Breathing': 0,
        'normal': 0,
        'Central Apnea': 1,
        'Hypopnea': 1,
        'Mixed Apnea': 1,
        'Obstructive Apnea': 1,
        'Noise': -1}



#

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
    db = DB()
    nwh_company = db.setup_by_project(Project.nwh)
    #bed_top = db.setup_by_mount(Mount.bed_top)
    invalid = db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
    setups = set(nwh_company) - set(invalid)

    sessions = set([db.session_from_setup(setup) for setup in setups])
    setups = [min(db.setups_by_session(session)) for session in sessions][3:]

    #setups = [6281]

    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

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
            reference = load_reference(sess, ['apnea'], db).fillna(method='ffill')
            setup_ref_path = os.path.dirname(db.setup_ref_path_npy(setup=sess, sensor=Sensor.natus, vs=VS.apnea))
            for file in os.listdir(setup_ref_path):
                #print(file)
                if 'chest.npy' in file:
                    abs_path = os.path.join(setup_ref_path, file)

                if ('annotation' in file or 'Annotation' in file) and '.csv' in file:
                    anno_path = os.path.join(setup_ref_path, file)
            ch = np.load(abs_path, allow_pickle=True)
            ann_df = pd.read_csv(anno_path)
        except:
            print(sess, "not ok")
            continue
        #reference = reference.asfreq('100ms', method='pad')
        #print(ann_df.keys())


        for k in ann_df.keys():
            if 'onset' in k or 'nset' in k:
                ann_df.rename(columns={k: 'onset'}, inplace=True)
            elif 'dur' in k or 'Dur' in k:
                ann_df.rename(columns={k: 'duration'}, inplace=True)
            elif 'dec' in k or 'Des' in k or 'cript' in k:
                ann_df.rename(columns={k: 'description'}, inplace=True)

        try:
            apnea_types = [s for s in np.unique(ann_df['description']) if 'pnea' in s]
        except:
            print("fail")
            continue
        ann_df.set_index("description", inplace=True)
        apneas_df = ann_df.loc[apnea_types]
        UP = 10
        DOWN = 512
        ch = sp.resample_poly(ch, UP, DOWN)
        plt.plot(ch)
        plt.show()
        apnea_ref_for_plot = np.zeros(len(ch))
        for _,row in apneas_df.iterrows():
            start_idx = int(row['onset']*UP)
            end_idx = int(start_idx + row['duration']*UP)
            apnea_ref_for_plot[start_idx:end_idx] = 1

        fs_ref = UP#get_fs(apneas_df)


        #continue
        #plt.close()
        #continue
        #respiration, fs_new = getSetupRespiration(sess)

        X = []
        y = []
        chunk_size_in_minutes = args.chunk
        time_chunk = fs_ref * chunk_size_in_minutes * 60
        step = fs_ref * args.step * 60
        print(time_chunk)
        print(step)


        X, y, valid = create_AHI_regression_training_data_NW_chest(respiration=ch, apnea_ref=apneas_df,
                                                time_chunk=time_chunk, step=step,
                                                scale=args.scale, fs=fs_ref)
        print(X.shape, y.shape)

        #plt.plot(2 * ch / max(ch))
        #plt.plot(apnea_ref_for_plot)
        # plt.title(f"setup id: {sess}")
        # create_dir(f'{SAVE_FOLDER}/{idx}/')
        # plt.savefig(f'{SAVE_FOLDER}/{idx}/{idx}_chest_apnea_with_gaps.png')
        #plt.show()


        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        np.save(os.path.join(save_path, str(sess) + '_valid.npy'), valid, allow_pickle=True)
        print("saved training data")
