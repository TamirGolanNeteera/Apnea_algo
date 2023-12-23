# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
from Tests.NN.create_apnea_count_AHI_data import delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, getApneaSegments

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *

db = DB()


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')

    return parser.parse_args()

STATUS_TO_CLASS = {'Empty':10, 'Motion':20, 'Running':30, 'Warming-up':40,'Low-Signal':50}
cols = {'Empty':'red', 'Motion':'blue', 'Running':'green', 'Warming-up':'gray','Low-Signal':'yellow'}

if __name__ == '__main__':

 #   args = get_args()


    sessions = [6842,6843, 6844]#6838, 6839, 6840, 6841]
    db = DB()

    for session in sessions:
        setups = db.setups_by_session(session)
        fig, ax = plt.subplots(len(setups), sharex=True, figsize=(10, 8))
        print(session, setups)
        for i,s in enumerate(setups):
            respiration, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(s)
            binss = np.repeat(bins, fs_new)
            ax[i].plot(I, label='I', linewidth=0.5)
            ax[i].plot(Q, label='Q', linewidth=0.5)

            p = db.setup_dir(s)
            print(p)
            pp = os.sep.join([p, 'NES_RES'])
            print(pp)
            nextdir = listdir(pp)[0]

            pp = os.sep.join([pp, nextdir])
            onlyfiles=listdir(pp)

            vs = [f for f in onlyfiles if 'usb_results' in f and 'stat' in f]
            if not vs:
                print("NO VS produced for session", s)
                continue
            vs=vs[0]
            stat = np.load(os.sep.join([pp,vs]), allow_pickle=True)
            print(np.unique(stat))

            stat_enum = []
            for k in stat:
                if k in STATUS_TO_CLASS.keys():
                    stat_enum.append(STATUS_TO_CLASS[k])
                else:
                    stat_enum.append(100)
            a = np.array(stat_enum)

            prev = 0
            splits = np.append(np.where(np.diff(a) != 0)[0],len(a)+1)+1

            for split in splits:
                b = np.arange(1,a.size+1,1)[prev:split]

                prev = split

                if len(b)==1 and b[0] == len(stat):
                    continue

                ax[i].axvspan(b[0] * fs_new, b[-1] * fs_new, color=cols[stat[b[0]]], alpha=0.1)
                ax[i].set_title(str(session)+' '+str(s) + ' '+db.setup_sn(s)[0], fontsize=9)
        plt.savefig(os.path.join('/Neteera/Work/homes/dana.shavit/', str(session)+'.png'))
        #plt.show()