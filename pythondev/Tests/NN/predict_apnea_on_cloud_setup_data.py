# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys

import numpy as np

sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch
from os import listdir
from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
import keras
from Tests.NN.create_apnea_count_AHI_data import compute_respiration, delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.create_apnea_count_AHI_data_regression_cloud_data_no_ref import get_empty_seconds_mb, create_AHI_regression_training_data_no_ref
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

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/apnea_2806/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/ahi_data_tamir/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/ahi_data_stiched/scaled/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/stitched_1712/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data_embedded_ds_filter/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data_embedded_orig_config_test/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data_tamir_wo_bug_test/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data_embedded_orig_config_test2/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_3/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_2/scaled/'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data_tamir_wo_bug/scaled/'
base_path  = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_2_chnage_valid/scaled'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_2_with_SW_model_filter_empty/scaled'
base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_2_with_SW_model/scaled'



fn = 'ahi' + '.png'
posture_class = {'Supine':0, 'Left':2, 'Right':3, 'Prone':1}
color_radar = {1:'red',2:'green', 3:'blue', 4:'magenta'}
setups = 'all'


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


device_map = {232:1, 238:1, 234:1, 236:1, 231:1,
                  240:2, 248:2, 254:2, 250:2, 251:2,
                  270:3, 268:3, 256:3, 269:3, 259:3,
                    278:4, 279:4, 273:4, 271:4, 274:4}
if __name__ == '__main__':

    res_dict = {}#'256':[], '273':[], '254':[], '234':[]}
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']


    if setups == 'all':
        setups = []
        setup_files = fnmatch.filter(os.listdir(base_path),'*_X.npy')
        label_files = fnmatch.filter(os.listdir(base_path),'*_y.npy')
        valid_files = fnmatch.filter(os.listdir(base_path),'*_valid.npy')
        ss_files = fnmatch.filter(os.listdir(base_path),'*_ss_ref*.npy')
        ss_pred_files = fnmatch.filter(os.listdir(base_path),'*_ss_pred*.npy')
        empty_files = fnmatch.filter(os.listdir(base_path),'*_empty*.npy')
        apnea_files = fnmatch.filter(os.listdir(base_path),'*_apnea*.npy')
        rank_files = fnmatch.filter(os.listdir(base_path),'*_radar_rank*.npy')
        qual_files = fnmatch.filter(os.listdir(base_path),'*_radar_quals*.npy')

        print("setup_files", setup_files, len(setup_files))

        for i, fn in enumerate(setup_files):
            sess = int(fn[0:fn.find('_')])
            setups.append(sess)

    print(len(setups),setups )
    dana_ahi_dict = {}
    yair_ahi_dict = {}
    p_dana = {}
    p_dana_valid = {}
    p_yair = {}
    y_true = []
    y_pred = []
    y_true_4class = []
    y_pred_4class = []
    y_true_2class = []
    y_pred_2class = []
    res = {}
    sessions_processed = []
    setups =  list(set([int(s)  for s in [fn[0:fn.find('_')] for fn in os.listdir('/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_3/scaled/')] if str.isdigit(s)]))
    db.update_mysql_db(0)
    # setups = db.setups_by_session(108148)
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

        session = db.session_from_setup(sess)
        # if session != 108186:
        #     continue
        rejected = [108146, 108152]
        if session in rejected:
            continue
#        fig, ax = plt.subplots(4, sharex=False, figsize=(14, 7))
#         try:
#
#             #respiration, fs_new, bins = getSetupRespirationCloudDB(sess)
#             ph = np.load(base_path+str(sess)+'_phase.npy')
#             phase_df = pd.DataFrame(ph)
#
#             phase_df.interpolate(method="linear", inplace=True)
#             phase_df.fillna(method="bfill", inplace=True)
#             phase_df.fillna(method="pad", inplace=True)
#
#             ph = phase_df.to_numpy()
#             respiration = compute_respiration(ph.flatten())
#
#             UP = 1
#             DOWN = 50
#             respiration = sp.resample_poly(respiration, UP, DOWN)
#             fs_new = int((db.setup_fs(sess) * UP) / DOWN)
#
#         except:
#             continue
        #print("setup length", len(respiration)/fs_new)
        # min_setup_length = 900
        # if len(respiration) / fs_new < min_setup_length:
        #     continue

        model_path = os.path.join('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/', 'NN', 'apnea')

        json_fn = '6284_model.json'
        hdf_fn = '6284_model.hdf5'
        json_file = open(os.path.join(model_path, json_fn), 'r')
        model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(model_json)
        model.load_weights(os.path.join(model_path, hdf_fn))
        model.compile(loss='huber_loss', optimizer=keras.optimizers.Adam())

        X_fn = [f for f in setup_files if str(sess) in f][0]
        y_fn = [f for f in label_files if str(sess) in f][0]
        v_fn = [f for f in valid_files if str(sess) in f][0]
        # e_fn = [f for f in empty_files if str(sess) in f][0]
        a_fn = [f for f in apnea_files if str(sess) in f][0]
        s_fn = [f for f in ss_files if str(sess) in f][0]
        p_fn = [f for f in ss_pred_files if str(sess) in f][0]
     ##   r_fn = [f for f in rank_files if str(session) in f][0]
        try:
            X = np.load(os.path.join(base_path, X_fn), allow_pickle=True)
            y = np.load(os.path.join(base_path, y_fn), allow_pickle=True)
            valid = np.load(os.path.join(base_path, v_fn), allow_pickle=True)
            ss = np.load(os.path.join(base_path, s_fn), allow_pickle=True)
            ss_pred = np.load(os.path.join(base_path, p_fn), allow_pickle=True)
            # empty = np.load(os.path.join(base_path, e_fn), allow_pickle=True)
            apnea = np.load(os.path.join(base_path, a_fn), allow_pickle=True)
            # rank = np.load(os.path.join(base_path, r_fn), allow_pickle=True)
            print(len(X), len(y), len(valid), len(ss))#, len(empty))


        except:
            print("failed to load setup data")
            continue

        session = db.session_from_setup(sess)
        print(session, sess)
        preds = model.predict(X).flatten()
        preds_int = np.stack([np.round(p,1) for p in preds])
        print(y)
        print(preds_int)
        #print(np.round(preds.flatten()[valid == 1]))
        pi=0
        time_chunk = 1200

        yy =[y[j] if valid[j] else 0 for j in range(len(y))]
        pp =[preds[j] if valid[j] else 0 for j in range(len(preds))]


        device_id = db.setup_sn(sess)[0]
        device_location = device_map[int(device_id) % 1000]
        # if device_location in [2,4]:
        #     continue


        ahi = np.mean(preds)*4

        sleep_perc = np.round(len(valid[valid == 1])/len(valid),2)

        if session not in res_dict.keys():
            res_dict[session] = {}

        #
        # duration = len(respiration)/36000

        num_apneas_from_y = np.sum(y)
        dana_from_y = 4*num_apneas_from_y/len(y)
        pahi_from_y =  num_apneas_from_y / (np.count_nonzero(ss)/3600)
        # yair_ahi_dict[session] = yair_ahi
        dana_ahi_dict[session] = pahi_from_y
        session_ahi_pdf_dict = {108139: 1.2, 108145: 10.8, 108186: 56.6, 108168: 37.6, 108146: 0.8, 108147: 6.1, 108148: 29.2, 108153: 5.4, 108154: 3.8, 108152: 16.2, 108170: 2.7, 108171: 13.7, 108175: 0.4, 108207: 5.5, 108191: 9.6, 108192: 4.1, 108201: 13.6, 108223: 8.9, 108298: 23.1, 108222: 5.2, 108226: 1.9, 108299: 11.8, 108303: 1.2, 108331: 3.7, 108348: 15.8, 108335: 9.8, 108349: 48.2}
        num_apneas_from_pred = np.sum(preds)
        pahi_from_pred = 4*num_apneas_from_pred/len(preds)

        len_valid = len(valid[valid == 1])
        # num_apneas_from_y_valid = np.sum(y[valid==1])
        # pahi_from_y_valid = 4*num_apneas_from_y_valid / len_valid
        num_apneas_from_pred_valid = np.sum(preds[valid==1])
        pahi_from_pred_valid = num_apneas_from_pred_valid / (len_valid / 4)
        pahi_by_ss_filter_empty = num_apneas_from_pred / (np.count_nonzero(np.logical_and(1 - ss_pred)) / 3600)

        print("#apneas", num_apneas_from_y, num_apneas_from_pred, pahi_from_y, pahi_from_pred)
        print("#apneas valid", num_apneas_from_pred_valid, pahi_from_pred_valid)
        p_dana[sess] = dana_from_y
        p_yair[sess] = pahi_from_y

        print(valid)
        res_dict[session][sess] = {'device': db.setup_sn(sess)[0][-3:],
        'pahi_from_y': pahi_from_y, 'pahi_from_pred': pahi_from_pred, 'pahi_from_pred_valid': pahi_from_pred_valid,
                              'pahi_by_ss' : pahi_by_ss_filter_empty}
        outfolder = base_path


        y_true.append(pahi_from_y)
        y_pred.append(pahi_by_ss_filter_empty)

        res[sess] = [pahi_from_y, pahi_by_ss_filter_empty]
        def ahi_class(ahi):
            if ahi < 5:
                return 0
            if ahi <= 15:
                return 1
            if ahi <= 30:
                return 2
            return 3

        y_true_4class.append(ahi_class(pahi_from_y))#pahi_from_y_valid))
        y_pred_4class.append(ahi_class(pahi_by_ss_filter_empty))
        y_true_2class.append([0 if ahi_class(pahi_from_y) <= 1 else 1])
        y_pred_2class.append([0 if ahi_class(pahi_by_ss_filter_empty) <= 1 else 1])


    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    plt.figure()
    cm4 = confusion_matrix(y_pred_4class, y_true_4class)
    cm2 = confusion_matrix(y_pred_2class, y_true_2class)

    TP = cm2[1][1]
    TN = cm2[0][0]
    FN = cm2[0][1]
    FP = cm2[1][0]
    if TP + FN == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    if TN + FP == 0:
        specificity = 0
    else:
        specificity = TN / (TN + FP)

    bx = sns.heatmap(cm4, annot=True, fmt="d", cmap="rocket")

    # Add labels and title
    bx.invert_yaxis()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('4-class Confusion Matrix')
    plt.savefig(os.path.join(base_path,"cm4_part.png"))
    # Show the plot
    plt.show()
    plt.close()
    plt.figure()
    bx = sns.heatmap(cm2, annot=True, fmt="d", cmap="rocket")
    bx.invert_yaxis()
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('2-class Confusion Matrix :: sensitivity=' + str(np.round(sensitivity, 2)) + " :: specificity=" + str(
        np.round(specificity, 2)))

    plt.savefig(os.path.join(base_path,"cm2_part.png"))
    plt.show()
    plt.close()
    device_map = {232: 1, 238: 1, 234: 1, 236: 1, 231: 1,
                  240: 2, 248: 2, 254: 2, 250: 2, 251: 2,
                  270: 3, 268: 3, 256: 3, 269: 3, 259: 3,
                  278: 4, 279: 4, 273: 4, 271: 4, 274: 4}

    plt.figure(figsize=(10,10))


    for device in range(1,5):
        y_true_2class = []
        y_pred_2class = []
        plt.figure(figsize=(10, 10))
        print("With Valid")
        for session, session_data in res_dict.items():
            print(session, session_data)

            for setup, setup_data in session_data.items():

                device_loc = device_map[int(setup_data['device'])]
                if device_loc > 0:
                    if device_loc != device:
                        continue
                try:
                    plt.xlim((0,60))
                    plt.ylim((0,60))
                    plt.scatter(setup_data['pahi_from_y'], setup_data['pahi_by_ss'], alpha=0.8, c=color_radar[device_loc], s=5)
                    y_true_2class.append([0 if ahi_class(setup_data['pahi_from_y']) <= 1 else 1])
                    y_pred_2class.append([0 if ahi_class( setup_data['pahi_by_ss']) <= 1 else 1])

                    plt.text(setup_data['pahi_from_y'], setup_data['pahi_by_ss'], str(device_loc) + ' ' + str(session), fontsize=6, alpha=0.6, c=color_radar[device_loc])
                except:
                    print(2)
        th = [5, 15, 30]
        for t in th:
            thick=0.5
            if t == 15:
                thick = 1
            plt.axhline(y=t, color='grey', linewidth=thick, alpha=thick)
            plt.axvline(x=t, color='grey', linewidth=thick, alpha=thick)
        plt.xlabel("gt")
        plt.ylabel("pred")
        if len(y_pred_2class) == 0:
            continue
        cm2 = confusion_matrix(y_pred_2class, y_true_2class)
        TP = cm2[1][1]
        TN = cm2[0][0]
        FN = cm2[0][1]
        FP = cm2[1][0]
        plt.title(f"MB2 AHI device {device} TP: {TP}/{(TP+FN)} ,TN: {TN}/{TN+FP}, TOTAL: {TP+TN}/{TP+TN+FN+FP}")

        plt.savefig(os.path.join(base_path, f'scatter_device_{device}_part.png'))
        plt.show()
        plt.close()
    print("y_pred", y_pred)
    print("y_true", y_true)
    print(res)
    # pd.DataFrame(res).T.rename_axis([']).to_csv(base_path + 'res.csv')
