# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
# from Tests.Plots.PlotRawDataRadarCPX import*
import argparse
import fnmatch
from os import listdir
import pickle
from sklearn import preprocessing
import pandas as pd
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from pptx import Presentation
from pptx.util import Inches, Cm
from io import BytesIO
from Tests.NN.create_apnea_count_AHI_data import MB_HQ, delays, getSetupRespirationCloudDBDebug, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebugDecimate, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, compute_respiration, compute_phase
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
import glob
import scipy.signal as sp


from scipy.signal import butter, filtfilt, lfilter, firwin

db = DB()

if __name__ == '__main__':
    MB_HQ = [109870, 109872, 109877, 109884, 109886, 109887, 109889, 109892, 109897
    , 109901, 109903, 109906, 109910, 109918, 109928, 109937, 109958, 109966, 110033
    , 110044, 110045, 110071, 110072, 110190, 110191, 110323, 110331, 110332, 110334
    , 110337, 110338, 110340, 110341, 110342, 110343, 110344, 110347, 110348, 110361
    , 110362, 110364, 110366, 110368, 110371, 110372, 110376, 110377, 110378, 110379
    , 110382, 110389, 110393, 110394, 110396, 110398, 110399, 110400, 110401, 110402
    , 110405, 110407, 110408, 110410, 110411, 110412, 110413, 110452, 110454]
    szmc_cloud_setups = [111358,111359, 111201,111202,111203,111204,111205,111206,111207,111208,111209,111210,111211,111212,111213,111220,111221,111293,111321,111322,111317,111328,111329,111330, ]
    herzog_cloud_setups = [110581,110771,110776,110782,110783,110804,110850,110851,110853,110871,110876,110883,110884,110940,110941,110999,111008,
                                111009,111064,111075,111080,111081,111267,111292,111309,111316,111319,111331,111332,111334,111335,111336,111340,111341,111476,111477,111478,111479,111480,111481]
    #setup_lists = [ db.benchmark_setups('mild_motion'), db.benchmark_setups('N130P_rest_benchmark'),db.benchmark_setups('N130P_ec_benchmark'), szmc_cloud_setups, herzog_cloud_setups, db.benchmark_setups('szmc_clinical_trials'), db.benchmark_setups('cen_exel')]
    setup_lists = [db.benchmark_setups('es_benchmark'), MB_HQ,  db.benchmark_setups('N130P_rest_benchmark'),szmc_cloud_setups, herzog_cloud_setups, db.benchmark_setups('cen_exel'), db.benchmark_setups('szmc_clinical_trials'), db.benchmark_setups('fae_rest')]
   # list_names = ['szmc_clinical_trials','cen_exel','mild_motion','N130P_rest_benchmark','N130P_ec_benchmark', 'szmc_cloud_setups', 'herzog_cloud_setups']
    hr_val_dict = {'es_benchmark':[],'MB':[],'N130P_rest_benchmark':[],'szmc_cloud_setups':[], 'herzog_cloud_setups':[], 'cen_exel':[], 'szmc_clinical_trials':[], 'fae_rest':[]}
    rr_val_dict = {'es_benchmark':[],'MB':[],'N130P_rest_benchmark':[],'szmc_cloud_setups':[], 'herzog_cloud_setups':[], 'cen_exel':[], 'szmc_clinical_trials':[], 'fae_rest':[]}

    hrv_dict = {'es_benchmark':[],'MB':[],'N130P_rest_benchmark':[],'szmc_cloud_setups':[], 'herzog_cloud_setups':[], 'cen_exel':[], 'szmc_clinical_trials':[], 'fae_rest':[]}
    rrv_dict = {'es_benchmark':[],'MB':[],'N130P_rest_benchmark':[],'szmc_cloud_setups':[], 'herzog_cloud_setups':[], 'cen_exel':[], 'szmc_clinical_trials':[], 'fae_rest':[]}

    setup_count = 0
    fig1, ax = plt.subplots(1,8, sharex=True, figsize=(45, 6))
    fig2, bx = plt.subplots(1,8, sharex=True, figsize=(45, 6))
    fig3, cx = plt.subplots(1,8, sharex=True, figsize=(45, 6))
    fig4, dx = plt.subplots(1,8, sharex=True, figsize=(45, 6))
    if True:#not os.path.isfile(os.path.join("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx", "rr_dict_big3.pkl")):
        for list_name, setups in enumerate(setup_lists):
            for i_sess, sess in enumerate(setups):
                db.update_mysql_db(sess)

                ts = None
                use_ts = False

                session = db.session_from_setup(sess)
                dist = db.setup_distance(sess)


                p = db.setup_dir(sess)
                raw_dir = os.sep.join([p, 'NES_RAW'])
                res_dir = os.sep.join([p, 'NES_RES'])

                try:
                    onlyfiles = listdir(res_dir)
                    if len(onlyfiles)==1:
                        res_dir = os.sep.join([res_dir, onlyfiles[0]])
                    print(res_dir)
                    onlyfiles = listdir(res_dir)
                    vs = [f for f in onlyfiles if 'VS' in f and 'csv' in f]
                    if len(vs) == 0:
                        vs = [f for f in onlyfiles if 'results' in f and 'csv' in f]
                    vs = vs[0]
                    try:
                        df = pd.read_csv(os.sep.join([res_dir, vs]))
                        setup_hr = []
                        for h in df['hr']:
                            if h > 0:
                                hr_val_dict[list(hr_val_dict.keys())[list_name]].append(h)
                                setup_hr.append(h)
                        setup_rr = []
                        for r in df['rr']:
                            if r > 0:
                                rr_val_dict[list(rr_val_dict.keys())[list_name]].append(r)
                                setup_rr.append(r)

                        rrv_dict[list(rr_val_dict.keys())[list_name]].append(np.var(setup_rr))
                        hrv_dict[list(hr_val_dict.keys())[list_name]].append(np.var(setup_hr))
                        print("ok")
                    except:
                        print("not ok")
                        continue
                    print(sess, "read vs file")
                    setup_count+=1
                except:
                    print(sess, "not ok")
                    continue
        print("done")
        # with open(os.path.join("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx", "hr_dict_big4.pkl"), 'wb') as fp:
        #     pickle.dump(hr_val_dict, fp)
        # with open(os.path.join("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx", "rr_dict_big4.pkl"), 'wb') as fp:
        #     pickle.dump(rr_val_dict, fp)
    # else:
    #     with open(os.path.join("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx", "hr_dict_big4.pkl"), 'rb') as fp:
    #         hr_val_dict = pickle.load(fp)
    #     with open(os.path.join("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx", "rr_dict_big4.pkl"), 'rb') as fp:
    #         rr_val_dict = pickle.load(fp)
    print("processed", setup_count, "setups")
    rr_bins = range(3,43)
    hr_bins = range(35,150,2)

    j = 0
    for k,v in rr_val_dict.items():
        print(k,len(v))
        if len(v) == 0:
            continue
        ax[j].axvline(x=20, linewidth=0.5)
        ax[j].hist(v, bins=rr_bins, alpha=0.3, label=k)
        ax[j].legend(loc='lower center', prop = { "size": 8 })

        j+=1
    ax[0].set_title("rr")
    j = 0
    for k,v in hr_val_dict.items():
        print(k,len(v))
        if len(v) == 0:
            continue
        bx[j].axvline(x=60, linewidth=0.5)
        bx[j].hist(v, bins=hr_bins, alpha=0.3, label=k)
        bx[j].legend(loc='lower center',prop = { "size": 8 })

        j+=1
    bx[0].set_title("hr")
    fig1.savefig("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx/rr.png")
    fig2 .savefig("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx/hr.png")

    j = 0
    for k, v in rrv_dict.items():
        print(k, len(v))
        if len(v) == 0:
            continue
        cx[j].hist(v, alpha=0.3, label=k)
        cx[j].legend(loc='lower center', prop={"size": 8})

        j += 1
    cx[0].set_title("rrv")
    j = 0
    for k, v in hrv_dict.items():
        print(k, len(v))
        if len(v) == 0:
            continue

        dx[j].hist(v, alpha=0.3, label=k)
        dx[j].legend(loc='lower center', prop={"size": 8})

        j += 1
    dx[0].set_title("hrv")
    fig3.savefig("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx/rrv.png")
    fig4.savefig("/Neteera/Work/homes/dana.shavit/Research/analysis/3.6.xx/hrv.png")
    plt.show()
