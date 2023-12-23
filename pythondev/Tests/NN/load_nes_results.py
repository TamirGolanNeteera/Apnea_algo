# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import time
from os import listdir
from os.path import isfile, join
#from OpsUtils.pylibneteera.ExtractPhaseFromCPXAPI import get_fs
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import count_apneas_in_chunk, compute_respiration
from pylibneteera.ExtractPhaseFromCPXAPI import get_fs
db = DB()

def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')

 return parser.parse_args()


if __name__ == '__main__':

 #   args = get_args()


    sessions = []
    db = DB('neteera_cloud_mirror')
    setups = db.all_setups()
    sn = [db.setup_sn(s) for s in setups]
    room = [db.setup_subject(s) for s in setups]
    sessions = [db.session_from_setup(setup) for setup in setups]
    MB_HQ = [109816, 109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897,
         109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966]
    #MB_HQ = [109889]
    #setups = [min(db.setups_by_session(session)) for session in sessions][3:]
    for i, s in enumerate(MB_HQ):
        print(i,s)
        #print(i, s, sn[i], room[i], db.setup_dir(s))
        p = db.setup_dir(s)
        print(s,p)

    l = []
    for i, s in enumerate(MB_HQ):

        #print(i, s, sn[i], room[i], db.setup_dir(s))
        p = db.setup_dir(s)

        raw_dir = os.sep.join([p, 'NES_RAW'])
        for file in os.listdir(raw_dir):
            if 'phase' in file and 'tmp' not in file:
                phase_fn = file
                path = os.path.join(raw_dir, phase_fn)
                ti_c = os.path.getctime(path)
                ti_m = os.path.getmtime(path)

                # Converting the time in seconds to a timestamp
                c_ti = time.ctime(ti_c)
                m_ti = time.ctime(ti_m)


        phase = np.load(os.sep.join([raw_dir, phase_fn]), allow_pickle=True)
        phase = compute_respiration(phase.to_numpy())
        plt.plot(phase)
        plt.title(str(m_ti))
        plt.show()
        continue
        pp = os.sep.join([p, 'NES_RES'])
        #print(p)
        if not os.path.isdir(pp):
            pp = os.sep.join([p, 'NES_RAW'])
            if not os.path.isdir(pp):
                print("NO res file for session", s)
                continue
        onlyfiles = listdir(pp)
        vs = [f for f in onlyfiles if '_VS' in f]
        if not vs:
            print("NO VS produced for session", s)
            continue
        vs=vs[0]
        df = pd.read_csv(os.sep.join([pp,vs]))

        rr = df['rr'].to_numpy()
        hr = df['hr'].to_numpy()

        perc_valid_rr = 100.0*(1.0 - len(rr[rr == -1]) / len(rr))
        perc_valid_hr = 100.0*(1.0 - len(hr[hr == -1]) / len(hr))
        print([s, len(df), perc_valid_hr, perc_valid_rr])
        #if len(df) > 7500:
        l.append([s, sessions[i], room[i][9:], int(sn[i][0][-5:]), len(df), perc_valid_hr, perc_valid_rr, 0.5*(perc_valid_hr+perc_valid_rr)])

    out_df = pd.DataFrame(l, columns=['Setup', 'Session','Room', 'Duration', 'SN',  'Valid HR %', 'Valid RR %', 'Mean_Perc'])
    out_df.to_csv('/Neteera/Work/homes/dana.shavit/analysis/mb_sessions_plus.csv')

    bins_df = pd.read_csv('/Neteera/Work/homes/dana.shavit/analysis/range_analysis.csv')
    bins_df.sort_values(by=['setup'], inplace=True)
    out_df.sort_values(by=['Setup'], inplace=True)

    vs_rate =  out_df['Mean_Perc'].to_numpy()
    bin3_rate = bins_df['bin_3'].to_numpy()/bins_df['seconds'].to_numpy()
    bin8_rate = bins_df['bin_8'].to_numpy() /bins_df['seconds'].to_numpy()
    plt.scatter(bin3_rate, vs_rate, s=3)
    plt.figure()
    plt.scatter(bin3_rate, vs_rate, s=3)
    plt.show()
    print("done")
