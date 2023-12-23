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

from Tests.NN.create_apnea_count_AHI_data import delays, getSetupRespirationCloudDBDebugWithTimestamp, MB_HQ, count_apneas_in_chunk, getSetupRespirationCloudDBDebug, getSetupRespirationLocalDBDebug, getSetupRespirationCloudDB, getApneaSegments
from Tests.NN.Chest.create_apnea_count_AHI_data_regression_MB_chest import apnea_class
import matplotlib.pyplot as plt
db = DB()
home = '/Neteera/Work/homes/dana.shavit/'
setups = [9715,9617,9713,9643]#[10216, 10217, 10228, 10229, 10232, 10233, 10249, 10250, 10253, 10254, 10271, 10272, 10291, 10292, 10295, 10296, 10303, 10304, 10311, 10312, 10192, 10193, 10188, 10189, 10204, 10205, 10208, 10209, 10220, 10221, 10224, 10225, 10236, 10237, 10245, 10246, 10275, 10276, 10279, 10280, 10283, 10284, 10287, 10288, 10299, 10300, 10307, 10308, 10315, 10316]
setups = [10425, 10426 ,10437, 10408, 10431, 10433, 10407]

base_path = '/Neteera/Work/homes/dana.shavit/analysis/vsm/3.5.11.1_issues_check_low_sig_before/hr_rr_stat_19_04_2023/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.1/long_ec2/'
setups = [4106,4176,4300,4514,4542,4774,4989,5090,5131,5997,6004,6985,7121,7571,8161]
setups=[8648,7815,4759,7507,4573,4746,4944,4959,6005,5090,4902,4856,7344,6969,4234,4774,4843,4281,4595,4868,6001,7609,7715,8143,5981,5989,4305,4106,4176,4300,4514,4542,4774,4989,5090,5131,5997,6004,6985,7121,7571,8161]
setups=[11109,11208]


base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/problem_radar_by_sn/'
base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/low_signal_study_fix2/'
setups = [11256, 11257, 11258, 11259, 11260, 11261, 11262, 11263]

base_path = '/Neteera/Work/homes/dana.shavit/Research/analysis/3.5.11.2/bins/'
setups = [10503,10515,10531,10551,10563,10587,10631,10651,10675,10695,10707,10727,10739,10759,10771,10803,10819,10835,10879,10919]


# setups_all = db.all_setups()
# setups1 = [s for s in setups_all if s > 9000]

# setups = []
# for s in setups1:
#     if db.setup_sn(s) and len(db.setup_sn(s)) >0 and db.setup_sn(s)[0]=='132021250209':
#         setups.append(s)

print(setups)
fn = 'bins_motion_test.png'
def get_args() -> argparse.Namespace:
 """ Argument parser
_change_filt
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


 return parser.parse_args()

def getSegments(v):
    #

    segments = []

    v_diff = np.diff(v, prepend=0, append=0)

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

if __name__ == '__main__':

    #args = get_args()


    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    print(setups)
    #setups = [109870]
    percentages = []
    mzc = []
    for i_sess, sess in enumerate(setups):
        print(db.setup_mount(sess), db.setup_target(sess), db.setup_posture(sess))
        print(i_sess,"/",len(setups))
        db.update_mysql_db(sess)
        fig, ax = plt.subplots(3, sharex=True , figsize=(7,4) )
        #fig, ax = plt.subplots(4, sharex=True  )
        perc=0.0

        ts = None
        use_ts = False
        if sess > 100000:
            #respiration, fs_new, bins, I, Q, = getSetupRespirationCloudDBDebug(sess)

            if use_ts:
                respiration, fs_new, bins, I, Q, ts = getSetupRespirationCloudDBDebugWithTimestamp(sess)
            else:
                respiration, fs_new, bins, I, Q = getSetupRespirationCloudDBDebug(sess)
            binss = np.repeat(bins, fs_new)

            # if ts is not None:
            #     ax[0].plot(ts, binss, label='bin')
            # else:
            #     ax[0].plot(binss, label='bin')
        else:
            respiration, fs_new, bins, I, Q, = getSetupRespirationLocalDBDebug(sess)

        #np.save(home+str(sess)+'_i.npy', I, allow_pickle=True)
        #np.save(home+str(sess)+'_q.npy', Q, allow_pickle=True)
        ax[0].set_title(str(sess)+' '+db.setup_subject(sess) + ' '+db.setup_sn(sess)[0]+' '+ str(db.setup_distance(sess))+' '+db.setup_posture(sess), fontsize=9)

        # ax[1].axvspan(tod[sess], tod[sess], color='red', alpha=1.0,
        #               label='TOD')
        if ts is not None:
            ax[0].plot(ts,  respiration, label='displacement', linewidth=0.5)
            #ax[2].plot(ts, I, label='I', linewidth=0.5)
            #ax[2].plot(ts, Q, label='Q', linewidth=0.5)

        else:

            #ax[0].plot(respiration, label='displacement')
            ia = np.round(np.max(I)-np.min(I))
            qa = np.round(np.max(Q)-np.min(Q))
            ax[0].plot(I, label='I', linewidth=0.5, color='blue', alpha=0.5)
            ax[1].plot(Q, label='Q', linewidth=0.5, color='blue', alpha=0.5)

            win_size=10*fs_new
            I_amp = np.zeros_like(I)
            Q_amp = np.zeros_like(I)

            for si in range(win_size, len(I), win_size):
                iamp = np.max(I[si-win_size:si])-np.min(I[si-win_size:si])
                qamp = np.max(Q[si-win_size:si])-np.min(Q[si-win_size:si])
                I_amp[si-win_size:si]=iamp
                I_amp[si-win_size:si]=iamp
                Q_amp[si-win_size:si]=qamp

            mi=np.round(np.mean(I_amp))
            mq=np.round(np.mean(Q_amp))
            #ax[2].plot(I_amp, label='I amplitude '+mi)
            #ax[2].plot(Q_amp, label='Q amplitude '+mq)
            #ax[2].axhline(y=2500, color='red', alpha=1)

            ax[0].legend(fontsize=9)
            ax[1].legend(fontsize=9)
            ax[2].legend(fontsize=9)
            #plt.title(str(sess)+" "+str(perc)+" "+db.setup_note(sess))
            plt.gcf().set_size_inches(10,7)
            #plt.tight_layout()
            ax[0].tick_params(axis='both', which='major', labelsize=6)
            ax[1].tick_params(axis='both', which='major', labelsize=6)
            ax[2].tick_params(axis='both', which='major', labelsize=6)
            ax[0].set_title('FIX ' + str(db.session_from_setup(sess))+" "+str(sess) + ' ' + db.setup_subject(sess) + ' ' + db.setup_sn(sess)[0] + ' ' + str(
                db.setup_distance(sess)) + ' ' + db.setup_posture(sess) +'\n'+"I,Q", fontsize=9)
            plt.savefig(home+str(sess)+'_'+db.setup_posture(sess)+'_'+fn)
            print("saved", home+''+str(sess)+fn)

            #plt.show()
            plt.close()
            #continue
        stat=None

        try:
            stat = np.array(np.load(os.path.join(base_path + 'dynamic', str(sess) + '_stat.data'), allow_pickle=True))
        except:
            print('no empty data for setup :-( ', sess)
        if stat is not None:
            #print(empty)
            empty_array = np.zeros(len(stat))
            print(len(stat[stat == 'empty chair']))
            empty_array[stat == 'empty chair'] = 1
            es = getSegments(empty_array)
            low_signal = np.zeros(len(stat))
            print(len(stat[stat == 'low_signal']))
            low_signal[stat == 'low_signal'] = 1
            running_array = np.zeros(len(stat))
            print(len(stat[stat == 'low_signal']))
            ls = getSegments(low_signal)
            running_array[stat == 'running'] = 1
            # us = getSegments(unknown_array)
            perc_ec = np.round(sum(empty_array) / len(empty_array), 2)
            print(sess, "Percentage EC", perc_ec)
            perc_ls = np.round(sum(low_signal) / len(low_signal), 2)
            print(sess, "Percentage LS", perc_ls)
            perc_ru = np.round(sum(running_array) / len(running_array), 2)
            print(sess, "Percentage RUN", perc_ru)
            percentages.append([sess, perc_ec, perc_ls, perc_ru])

            for e in es:
                ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='red', alpha=0.1)
                ax[1].axvspan(e[0] * fs_new, e[1] * fs_new, color='red', alpha=0.1)
                ax[2].axvspan(e[0] * fs_new, e[1] * fs_new, color='red', alpha=0.1)

            for e in ls:
                ax[0].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.1)
                ax[1].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.1)
                ax[2].axvspan(e[0] * fs_new, e[1] * fs_new, color='green', alpha=0.1)

        stat_qual = None
        try:
            rr_amp_qual = np.array(
                np.load(os.path.join(base_path + 'accumulated', str(sess) + '_zc.npy'), allow_pickle=True))
            print("len(stat)", len(rr_amp_qual))
            print(np.mean(rr_amp_qual))
            if stat is not None:
                empty_for_pzc = empty_array[len(empty_array)-len(rr_amp_qual):]
                mean_pzc = np.round(np.mean(rr_amp_qual[empty_for_pzc == 0]), 2)
            else:
                mean_pzc = np.round(np.mean(rr_amp_qual), 2)
            mzc.append([sess, mean_pzc])
            rr_amp_qual = np.repeat(rr_amp_qual, fs_new)
            ax[3].set_title('ZC percentile (Mean = ' + str(mean_pzc) + ')', fontsize=9)
            ax[3].plot(rr_amp_qual, color='blue', label='percentile zc', alpha=0.5, linewidth=0.5)

            es = getSegments(empty_for_pzc)

            perc = np.round(sum(empty_array) / len(empty_array), 2)
            print(sess, "Percentage EC", perc)
            percentages.append([sess, perc])

            for e in es:
                ax[3].axvspan(e[0] * fs_new, e[1] * fs_new, color='red', alpha=0.1)
            # ax[4].set_yticks(range(-1,5,1))
        except:
            print('no pzc data for setup', sess)



        rr_qual = None
        try:
            rr_amp_qual = np.array(
                np.load(os.path.join(base_path + 'accumulated', str(sess) + '_zc.npy'), allow_pickle=True))
            print("len(rr_qual)", len(rr_amp_qual))
            #rr_amp_qual = np.repeat(rr_amp_qual, fs_new)
            zc = np.stack(rr_amp_qual)
            ax[0].set_title('FIX ' + str(db.session_from_setup(sess))+" "+str(sess) + ' ' + db.setup_subject(sess) + ' ' + db.setup_sn(sess)[0] + ' ' + str(
                db.setup_distance(sess)) + ' ' + db.setup_posture(sess) +'\n'+"I,Q", fontsize=9)
            #ax[0].set_title("Zero Crossing Bin Score")
            for pl_i in range(zc.shape[1]):
                ax[0].plot(zc[:,pl_i], label=str(pl_i), linewidth=0.5)
            #ax[1].plot(rr_amp_qual, color='magenta', label='rr_qual', alpha=0.5)
            # ax[4].set_yticks(range(-1,5,1))
        except:
            print('no zc bin score  data for setup', sess)
        rr_qual = None
        try:
            rr_amp_qual = np.array(
                np.load(os.path.join(base_path + 'accumulated', str(sess) + '_zc.npy'), allow_pickle=True))
            print("len(rr_qual)", len(rr_amp_qual))
            #rr_amp_qual = np.repeat(rr_amp_qual, fs_new)
            zc = np.stack(rr_amp_qual)
            ax[1].set_title("AMP", fontsize=9)
            for pl_i in range(zc.shape[1]):
                ax[1].plot(zc[:,pl_i], label=str(pl_i), linewidth=0.5)
            #ax[1].plot(rr_amp_qual, color='green', label='rr_amp_qual', alpha=0.5)
            # ax[4].set_yticks(range(-1,5,1))
        except:
            print('no bin var data for setup', sess)
        cols = ['lightcoral', 'deepskyblue', 'indianred', 'skyblue', 'brown', 'steelblue', 'firebrick',
                'dodgerblue', 'salmon', 'cornflowerblue', 'tomato', 'royalblue', 'orangered', 'blue', 'red',
                'mediumblue']

        # try:
        #     rr_amp_qual = np.array(
        #         np.load(os.path.join(base_path + 'accumulated', str(sess) + '_mean.npy'), allow_pickle=True))
        #
        #     zc = np.stack(rr_amp_qual)
        #     ax[1].set_title("Bin Mean Score", fontsize=9)
        #     for pl_i in range(zc.shape[1]):
        #         ax[1].plot(zc[:,pl_i], label=str(pl_i))#, color=cols[pl_i])
        #     #ax[1].plot(rr_amp_qual, color='green', label='rr_amp_qual', alpha=0.5)
        #     # ax[4].set_yticks(range(-1,5,1))
        # except:
        #     print('no rr amp qual data for setup', sess)
        rr_amp_qual = None
        try:
            rr_amp_qual = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc.npy'), allow_pickle=True))
            print("len(rr_amp_qual)", len(rr_amp_qual))
            #rr_amp_qual = np.repeat(rr_amp_qual, fs_new)
            zc = np.stack(rr_amp_qual)
            ax[2].set_title("ZC", fontsize=9)
            for pl_i in range(zc.shape[1]):
                ax[2].plot(zc[:,pl_i], label=str(pl_i))#,color=cols[pl_i])
            #ax[3].plot(np.argmax(zc, axis=1), label='selected bin')

            #ax[3].plot(rr_amp_qual, color='blue', label='q amp', alpha=0.5)
            #ax[4].set_yticks(range(-1,5,1))
        except:
            print('no reject data for setup', sess)

        reason=None
        try:
            reason = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_reject_reason.npy'), allow_pickle=True))
            reason = np.repeat(reason, fs_new)

            #ax[3].plot(reason, color='magenta', label='reason')
            #ax[3].set_yticks(range(-1,5,1))
        except:
            print('no reject data for setup', sess)
        zc=None
        try:
            zc = np.array(np.load(os.path.join(base_path+'accumulated', str(sess)+'_zc.npy'), allow_pickle=True))
            zc = np.repeat(zc, fs_new)
            zc = np.hstack([np.zeros(len(respiration)-len(zc)), zc])
            ax[2].plot(zc, color='red', label='zc', linewidth=0.5)
            ax[2].axhline(y=0.28, color='blue', alpha=0.5)
            #ax[2].axhline(y=0.4, color='blue', alpha=0.5)

            zc = np.array(np.load(os.path.join(base_path + 'accumulated', str(sess) + '_amp.npy'),
                                  allow_pickle=True))
            zc = np.repeat(zc, fs_new )
            zc = np.hstack([np.zeros(len(respiration)-len(zc)), zc])
            ax[1].plot(zc, color='red', label='amp', linewidth=0.5)
            #ax[1].axhline(y=500, color='blue', alpha=0.5)
            ax[1].axhline(y=5000, color='blue', alpha=0.5)

        except:
            print('no zc data for setup', sess)


        ts_str = '_'
        if ts is not None:
            ts_str = '_ts_'

        #ax[1].legend()
        ###ax[3].axhline(y=int(0.5 + db.setup_distance(sess)/130), color='red', alpha=0.5)
        # ax[1].axhline(y=0.35, color='red', alpha=0.5)
        #ax[2].legend()
#        ax[3].axhline(y=0.28, color='red', alpha=0.5, linewidth=0.5)
#        ax[3].axhline(y=0.5, color='magenta', alpha=0.5, linewidth=0.5)
        #ax[3].legend()
        #ax[4].legend()
        #ax[5].legend()
        ax[2].legend(fontsize=9)
        ax[0].legend(fontsize=9)
        ax[1].legend(fontsize=9)
        #plt.title(str(sess)+" "+str(perc)+" "+db.setup_note(sess))
        plt.gcf().set_size_inches(10,7)
        #plt.tight_layout()
        ax[0].tick_params(axis='both', which='major', labelsize=6)
        ax[1].tick_params(axis='both', which='major', labelsize=6)
        ax[2].tick_params(axis='both', which='major', labelsize=6)
       # ax[3].tick_params(axis='both', which='major', labelsize=6)
        plt.savefig(home+str(sess)+'_'+db.setup_posture(sess)+'_'+fn)
        print("saved", home+str(sess)+fn)

        plt.show()
        plt.close()
    print(percentages)
    print(mzc)
    p = np.stack(percentages)
    #np.save(home+'perc_after.npy', p[:,1], allow_pickle=True)
    print("done")

    stats = {}
    stats['13'] = {}
    stats['21'] = {}
    stats['13']['Lying'] = []
    stats['13']['Sitting'] = []
    stats['21']['Lying'] = []
    stats['21']['Sitting'] = []

    for m in mzc:
        radar = db.setup_sn(m[0])[0][0:2]

        posture = db.setup_posture(m[0])
        stats[radar][posture].append(m[1])

    print("130 lying",np.mean(stats['13']['Lying']))
    print("210 lying",np.mean(stats['21']['Lying']))
    print("130 Sitting",np.mean(stats['13']['Sitting']))
    print("210 Sitting",np.mean(stats['21']['Sitting']))