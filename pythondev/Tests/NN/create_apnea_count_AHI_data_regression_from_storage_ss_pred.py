# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
from pathlib import Path

from Tests.NN.tamir_setups_stitcher import stitch_and_align_setups_of_session

sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch
import scipy.signal as sp
from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import delays, MB_HQ, count_apneas_in_chunk,compute_respiration
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class

db = DB()
session_ahi_pdf_dict = {108139: 1.2, 108145: 10.8, 108186: 56.6, 108168: 37.6, 108146: 0.8, 108147: 6.1, 108148: 29.2, 108153: 5.4, 108154: 3.8, 108152: 16.2, 108170: 2.7, 108171: 13.7, 108175: 0.4, 108207: 5.5, 108191: 9.6, 108192: 4.1, 108201: 13.6, 108223: 8.9, 108298: 23.1, 108222: 5.2, 108226: 1.9, 108299: 11.8, 108303: 1.2, 108331: 3.7, 108348: 15.8, 108335: 9.8, 108349: 48.2}


def create_AHI_regression_training_data_MB_phase_with_sleep_ref(respiration, apnea_ref, sleep_ref, empty_ref, time_chunk, step, scale, fs):
    print("in")
    X = []
    y = []
    valid = []
    apnea_segments = []
    if apnea_ref is not None:
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

            apnea_segments.append([start_idx, end_idx, apnea_duration[a_idx], 1])
        print(len(apnea_segments), "apneas in setup")

    for i in range(time_chunk, len(respiration), step):
        #print(i, "phase", len(respiration)/fs, "sleep", len(sleep_ref), "apnes", len(apnea_ref))
        v = 1

        seg = respiration[i - time_chunk:i]
        start_fs = int((i - time_chunk)/fs)
        end_fs = int(i/fs)


        if len(seg) != time_chunk:
            continue

        if sleep_ref is not None:
            if start_fs > len(sleep_ref) or end_fs > len(sleep_ref):
                v = 0
            else:
                sleep_labels_in_chunk = sleep_ref[start_fs:end_fs]
                #print(len(sleep_labels_in_chunk[sleep_labels_in_chunk != 1])/len(sleep_labels_in_chunk))
                if len(sleep_labels_in_chunk[sleep_labels_in_chunk != 1])/len(sleep_labels_in_chunk) > 0.5:
                    #print(v,"wake")
                    v = 0
        # if empty_ref is not None:
        #     if start_fs > len(empty_ref) or end_fs > len(empty_ref):
        #         v = 0
        #     else:
        #         empty_labels_in_chunk = empty_ref[start_fs:end_fs]
        #         #print(len(empty_labels_in_chunk[empty_labels_in_chunk != 1])/len(empty_labels_in_chunk))
        #         if len(empty_labels_in_chunk[empty_labels_in_chunk != 1])/len(empty_labels_in_chunk) > 0.75:
        #             #print(v,"wake")
        #             v = 0
                    
        if scale:
            X.append(preprocessing.scale(seg))
        else:
            X.append(seg)

        num_apneas = 0 if apnea_ref is None else count_apneas_in_chunk(start_t=start_fs, end_t=end_fs, apnea_segments=apnea_segments)

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


def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 parser.add_argument('-load_path', metavar='Location', type=str, required=True, help='location of stitched data')
 parser.add_argument('-save_path', metavar='Location', type=str, required=True, help='location of saved data')
 parser.add_argument('--scale', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--show', action='store_true', help='Scale data to m=0 s=1')
 parser.add_argument('--overwrite', action='store_true',  required=False, help='Overwrite existing output')
 parser.add_argument('--pred_sleep', action='store_true',  required=False, help='calc sleep stages by NN predictions and not gt')
 parser.add_argument('--filter_empty', action='store_true',  required=False, help='Filter empty from input')
 parser.add_argument('-chunk', metavar='window', type=int, required=True, help='signal size')
 parser.add_argument('-step', metavar='window', type=int, required=True, help='stride for signal creation')


 return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    if args.scale:
        save_path = os.path.join(args.save_path, 'scaled')
    else:
        save_path = os.path.join(args.save_path, 'unscaled')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    db = DB('neteera_cloud_mirror')
    setups = db.setup_by_sensor(Sensor.respironics_alice6)
    sessions = list(set(
        [db.session_from_setup(setup) for setup in setups if setup > 112000 and setup != 112241 and setup != 112242]))


    for i_sess, sess in enumerate(sessions):
        # try:
        #     if sess != 108171:
        #         continue
            phase_dir = None
            phase_dir = '/Neteera/Work/homes/tamir.golan/embedded_phase/MB_old_and_new_model_2_311223/hr_rr_ra_ie_stat_intra_breath_31_12_2023/accumulated'
            fs = 10 if phase_dir else 500
            try:
                stitch_dict = stitch_and_align_setups_of_session(sess, phase_dir=phase_dir)
            except:
                continue
            for setups in list(stitch_dict):
                print(f'process {setups}')
                stat_data = stitch_dict[setups]['stat']
                if args.pred_sleep:
                    ss_ref = None
                    for setup in setups:
                        try:
                            ss_pred = np.load(f'/Neteera/Work/homes/tamir.golan/API_plots/{setup}.npy',
                                             allow_pickle=True)
                            break
                        except FileNotFoundError:
                            pass

                ss_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*sleep*'))[0], allow_pickle=True)
                apnea_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*pnea*'))[0], allow_pickle=True)
                sp02 = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*SpO2*'))[0], allow_pickle=True)[::50]
                print(sess, ' apneas ', str(len(ss_ref) > len(apnea_ref)))
                hr_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*HR*'))[0], allow_pickle=True)
                if stitch_dict[setups]['ref_earlier']:
                    ss_ref = ss_ref[stitch_dict[setups]['gap'] // fs:]
                    apnea_ref = apnea_ref[stitch_dict[setups]['gap'] // fs:]
                    sp02 = sp02[stitch_dict[setups]['gap'] // fs:]



                for setups in list(stitch_dict.keys()):
                    setup = setups[0]
                    device = stitch_dict[setups]['device_loc']
                    phase_df = pd.DataFrame(stitch_dict[setups]['phase'])

                    phase_df.interpolate(method="linear", inplace=True)
                    phase_df.fillna(method="bfill", inplace=True)
                    phase_df.fillna(method="pad", inplace=True)

                    phase = phase_df.to_numpy().flatten()

                    if phase_dir is None:
                        respiration = compute_respiration(phase)
                        UP = 1
                        DOWN = 50
                        respiration = sp.resample_poly(respiration, UP, DOWN)
                    else:
                        respiration = phase
                    fs_new = 10



                    print(len(phase)/500, len(apnea_ref), len(ss_ref), np.unique(apnea_ref))


                    print(":::::::: processing setup", setup)
                    if args.overwrite and os.path.isfile(os.path.join(save_path,str(setup) + '_X.npy')):
                        print(setup, "done, skipping")
                        continue

                    print(np.unique(apnea_ref))
                    apnea_ref_class = np.zeros(len(apnea_ref))

                    for i in range(len(apnea_ref)):
                        #print(i, apnea_ref[i], apnea_ref[i] in apnea_class.keys())
                        if apnea_ref[i] not in apnea_class.keys():
                            apnea_ref_class[i] = -1
                        else:
                            apnea_ref_class[i] = int(apnea_class[apnea_ref[i]])

                    print(np.unique(apnea_ref_class))

                    print(np.unique(ss_ref))
                    ss_ref_class = np.zeros(len(ss_ref))
                    ss_class = {'N1':1, 'N2':1, 'N3':1, 'W':0, 'R':1}
                    for i in range(len(ss_ref)):
                        # print(i, ss_ref[i], ss_ref[i] in ss_class.keys())
                        if ss_ref[i] not in ss_class.keys():
                            ss_ref_class[i] = -1
                        else:
                            ss_ref_class[i] = int(ss_class[ss_ref[i]])

                    print(np.unique(ss_ref_class))

                    empty_ref_class = np.zeros(len(stat_data))
                    #print(np.unique(empty_ref))
                    full_statuses = ['FULL', 'MOTION', 'LOW_SIGNAL']
                    for i in range(len(stat_data)):
                        if stat_data[i] == 'EMPTY':
                            empty_ref_class[i] = 0
                        else:
                            empty_ref_class[i] = 1

                    #respiration, fs_new, bins = getSetupRespirationCloudDB(sess)

                    # min_setup_length = 15000
                    # if len(respiration)/fs_new < min_setup_length:
                    #     continue

                    chunk_size_in_minutes = args.chunk
                    time_chunk = fs_new * chunk_size_in_minutes * 60
                    step = fs_new * args.step * 60

                    try:
                        X, y, valid,  = create_AHI_regression_training_data_MB_phase_with_sleep_ref(respiration=respiration,
                                                                                    apnea_ref=apnea_ref_class,
                                                                                    sleep_ref=ss_ref_class,
                                                                                    empty_ref=empty_ref_class,
                                                                                    time_chunk=time_chunk,
                                                                                    step=step,
                                                                                    scale=args.scale,
                                                                                    fs=fs_new)
                        print("np.unique(y)",np.unique(y), setup, db.session_from_setup(setup))
                        print(y)
                    except:
                        print("oh crap")
                    #continue
                    print(X.shape, y.shape)
                    print("successfully created AHI labels")
                    np.save(os.path.join(save_path,str(setup) + '_y.npy'), y, allow_pickle=True)
                    np.save(os.path.join(save_path,str(setup) + '_X.npy'), X, allow_pickle=True)
                    np.save(os.path.join(save_path, str(setup) + '_valid.npy'), valid, allow_pickle=True)
                    np.save(os.path.join(save_path, str(setup) + '_empty_ref_class.npy'), empty_ref_class, allow_pickle=True)
                    np.save(os.path.join(save_path, str(setup) + '_ss_ref_class.npy'), ss_ref_class, allow_pickle=True)
                    if args.pred_sleep:
                        np.save(os.path.join(save_path, str(setup) + '_ss_pred_class.npy'), ss_pred, allow_pickle=True)
                    np.save(os.path.join(save_path, str(setup) + '_apnea_ref_class.npy'), apnea_ref_class, allow_pickle=True)
                print("saved training data")
