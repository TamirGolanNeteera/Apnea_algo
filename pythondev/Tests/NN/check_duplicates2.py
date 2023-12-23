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

from Tests.NN.create_apnea_count_AHI_data import delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebugDecimate, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
import glob
import scipy.signal as sp

db = DB()
home = '/Neteera/Work/homes/dana.shavit/Research/analysis/'

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/matress_search/'

setups = np.arange(110595, 111300)


def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')
 return parser.parse_args()

if __name__ == '__main__':
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    print(setups)

    clip = True

    #setups = [109870]
    d = []
    setups_with_dupes = {}
    sns_with_dupes = {}
    setups_no_dupes = {}
    sns_no_dupes = {}
    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(2, sharex=True, figsize = (20, 10))
        setup_row = []

        setup_row.append(sess)
        setup_row.append(db.setup_duration(sess))
        try:
            setup_row.append(db.setup_sn(sess)[0])
        except:
            setup_row.append(0)

        ts = None
        use_ts = False

        if db.mysql_db == 'neteera_cloud_mirror':
            try:
                sig, fs_new, bins, I, Q = getSetupRespirationCloudDBDebugDecimate(sess)
            except:
                continue
        else:
            try:
                sig, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(sess)
            except:
                continue

        binss = np.repeat(bins, fs_new)

        if clip:
            clip_len = 14000 * fs_new
            I = I[:clip_len]
            Q = Q[:clip_len]
            binss = binss[:clip_len]

        ax[0].plot(I, label='I', linewidth=0.25)
        ax[0].plot(Q, label='Q', linewidth=0.25)
        ax[1].plot(binss, label='bins', linewidth=0.5)

        nsec = int(np.floor(len(I)/fs_new))
        new_len = int(nsec*fs_new)
        I = I[:new_len]
        sig_seconds = I.reshape(nsec, fs_new)

        firstsec = 0
        lastsec = sig_seconds.shape[0]
        dupes = []

        if not clip:
            for i in range(firstsec, lastsec):
                if i % 1000 == 0:
                    print(i, "/", lastsec)
                pattern1 = sig_seconds[i, 0:10]
                p1 = np.tile(pattern1, [nsec,1])

                diff = np.sum(np.abs(sig_seconds[:,0:10]-p1), axis=1)

                zero_idxs = np.where(diff < 0.000001)[0]

                if len(zero_idxs)>1:
                    #np.delete(zero_idxs, np.where(zero_idxs == i))
                    dupes.append(zero_idxs)

                    for z in zero_idxs:
                        ax[0].axvspan((z*fs_new), (z+1)*fs_new, color='magenta', alpha=0.2)
                        ax[1].axvspan((z*fs_new), (z+1)*fs_new, color='magenta', alpha=0.2)
        sn = int(db.setup_sn(sess)[0])
        print(dupes)
        unique_list = []

        print("UNIQUE", unique_list)
        print(sess, sn,"**********************************")
        # for i in range(1, len(dupes), 3):
        #     print(dupes[i][0]-dupes[i-3][0])
        if len(dupes)>0:
            setups_with_dupes[sess] = dupes
            if sn not in sns_with_dupes.keys():
                sns_with_dupes[sn] = []
            sns_with_dupes[sn].append([sess, len(dupes)])
        else:
            if sn not in sns_no_dupes.keys():
                sns_no_dupes[sn] = []
            sns_no_dupes[sn].append([sess, len(dupes)])
            setups_no_dupes[sess] = dupes
        ax[0].set_title(str(sess)+' '+str(sn)+' @'+str(fs_new)+' Hz')
        #plt.show()
        plt.savefig(base_path+str(sess)+'_'+'dupes.png')
        plt.close()
    print("setups with duplicates")
    for k,v in setups_with_dupes.items():
        print(k,len(v))
    print("devices with duplicates")
    for k,v in sns_with_dupes.items():
        print(k,v)
    print("setups without duplicates")
    for k, v in setups_no_dupes.items():
        print(k, len(v))
    print("devices without duplicates")
    for k, v in sns_no_dupes.items():
        print(k, v)