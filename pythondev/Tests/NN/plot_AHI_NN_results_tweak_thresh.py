import matplotlib.pyplot as plt
from Tests.vsms_db_api import DB
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import fnmatch
import os
import pandas as pd

col = list(mcolors.cnames.keys())
colors = ['tan', 'lightcoral', 'red', 'green', 'blue', 'cyan', 'magenta', 'gold', 'darkgreen', 'yellow', 'purple',
          'lightblue', 'gray', 'skyblue', 'pink', 'mediumslateblue', 'lightseagreen', 'coral']

# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/3009_full_nwh_full_mb__workable_scaled/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/3009_full_nwh_full_mb_workable_scaled'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0510_full_nwh_full_mb_workable_scaled/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/2109_full_nwh_full_mb_scaled_aug'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/2109_full_nwh_full_mb_scaled_remove_problem_setups'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0809_fe_nwh_mb50_scaled/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/07_10_back_to_HQ/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/07_10_back_to_HQ/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/1410_new_script_old_data_0409_fe_nwh_mb50/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0409_fe_nwh_mb50/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/1610_full_nwh_full_mb_workable_scaled_undersamle_0'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/1810_no_empty'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/2610_no_empty_aug'#
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/3110_no_empty_aug'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0809_fe_nwh_mb50_segmentation'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0611_chest_2/'

# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/test_chest_with_and_without/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/test_with_chest/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/test_chest_without2/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/280523_test/'
lp = '//Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/3105_new_generator_w_aug/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/3105_new_generator_hour2/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/280523_test_20/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/280523_test_p30_bs256/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/ahi_data_embedded/'
# lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/0908_test_resnet_on_10hz/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/115_ahi_data_embedded/'
lp = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/trained_nns/with_mb2/'
print('Processing', lp)
# NN self test
self_test_pred_files = fnmatch.filter(os.listdir(lp), '*self_test*_pred.npy')
self_test_label_files = fnmatch.filter(os.listdir(lp), '*self_test*_gt.npy')
self_test_valid_files = fnmatch.filter(os.listdir(lp), '*self_test*_valid.npy')
print(self_test_pred_files)

pred_files = fnmatch.filter(os.listdir(lp), '*_pred.npy')
label_files = fnmatch.filter(os.listdir(lp), '*_gt.npy')
valid_files = fnmatch.filter(os.listdir(lp), '*_valid.npy')
bad_files = fnmatch.filter(os.listdir(lp), 'bad*.npy')
pred_files = [p for p in pred_files if 'self' not in p]
label_files = [p for p in label_files if 'self' not in p]
valid_files = [p for p in valid_files if 'self' not in p]

setup_maes = {}
setup_maxes = {}
res = {}
res_v = {}
pref_radar_res = {}
other_radar_res = {}
pref_radar_res_v = {}
other_radar_res_v = {}

gt_all_class = []
pred_all_class = []
gt_all_class_valid = []
pred_all_class_valid = []

all_gt = []

two_class_outputs_by_threshold = []

# l.append([s, sessions[i], room[i][9:], int(sn[i][0][-5:]), len(df), perc_valid_hr, perc_valid_rr, 0.5*(perc_valid_hr+perc_valid_rr),  100*(lowrr/len(df)), 100*(empty/len(df)), 100*(running/len(df))])
#
#         print(s, int(db.setup_sn(s)[0]), db.setup_mount(s))
#         #print(radars_and_perc.keys())
#
#         if int(sn[i][0][-5:]) not in radars_and_perc.keys():
#             radars_and_perc[int(sn[i][0][-5:])] = []
#             radars_and_installation[int(sn[i][0][-5:])] =db.setup_mount(s)
#             radars_and_facilities[int(sn[i][0][-5:])] = db.setup_subject(s)
#         radars_and_perc[int(sn[i][0][-5:])].append(0.5*(perc_valid_hr+perc_valid_rr))
#     out_df = pd.DataFrame(l, columns=['Setup', 'Session', 'Room','SN', 'Duration',   'Valid HR %', 'Valid RR %', 'Mean_Perc',  'lowrr_perc',  'empty_perc','running_perc'])

pref_radars = [40438, 40364, 40387, 40400, 40440, 40380, 40422, 40393, 40445]
bm = db.setup_nwh_benchmark()

for i, fn in enumerate(label_files):

    setup_str = fn[0:fn.find('_')]
    setup = int(setup_str)

    # print(setup)
    gt = np.load(os.path.join(lp, fn), allow_pickle=True)
    pred = np.load(os.path.join(lp, setup_str + '_pred.npy'), allow_pickle=True)
    #
    if np.isnan(gt).any() or np.isnan(pred).any():
        continue

    valid = np.ones_like(pred)  # np.load(os.path.join(lp, setup_str + '_valid.npy'), allow_pickle=True)
    print(gt)
    print(np.round(pred))
    print(valid)
    # print(setup, len(valid[valid==1])/len(valid),"%", valid)

    all_gt.append(gt)
    # print(gt)
    # print(np.round(pred))
    # if setup > 100000:
    #     print(setup, np.mean(np.abs(pred - gt)))

    plt.figure()
    mrkr = "^" if setup > 100000 else "o"

    # print(setup)
    s = 0

    for j, v in enumerate(gt):

        s += int(np.abs(v - pred[j]))
        alpha = 1.0
        if j > len(valid) - 1:
            continue
        if valid[j]:
            sz = 24
            bc = 'black'
            alpha = 1.0
        else:
            sz = 10
            bc = 'black'
            alpha = 0.3
        plt.title(setup_str)

        plt.scatter(v, pred[j], c=bc, s=26, marker=mrkr, alpha=alpha)
        plt.scatter(v, pred[j], c=colors[setup % len(colors)], s=sz, label=setup_str, marker=mrkr, alpha=alpha)
        plt.annotate(str(int(v)) + ' ' + str(int(pred[j])) + ' ' + str(valid[j]), (v, pred[j]), fontsize=6, alpha=alpha)

    seg_len_in_min = 15

    factor = 60 / seg_len_in_min

    sleep_time = len(gt) * factor
    ahi_pred = sum(pred) / (len(pred) / factor)
    ahi_gt = sum(gt) / (len(gt) / factor)
    gt_v = gt[valid == 1]
    pr_v = pred[valid == 1]
    print(setup, ahi_gt, ahi_pred)

    ahi_pred_v = sum(pr_v) / (len(pr_v) / factor)
    ahi_gt_v = sum(gt_v) / (len(gt_v) / factor)
    db.update_mysql_db(setup)
    sn = int(db.setup_sn(setup)[0][-5:])
    print(setup, ahi_gt_v, ahi_pred_v)
    if sn in pref_radars or setup in bm:
        pref_radar_res[setup] = [ahi_gt, ahi_pred]
        pref_radar_res_v[setup] = [ahi_gt_v, ahi_pred_v]
    else:
        other_radar_res[setup] = [ahi_gt, ahi_pred]
        other_radar_res_v[setup] = [ahi_gt_v, ahi_pred_v]

    res[setup] = [ahi_gt, ahi_pred]
    res_v[setup] = [ahi_gt_v, ahi_pred_v]
    fn = setup_str + '_segments_seg.png'
    plt.savefig(os.path.join(lp, fn))
    plt.close()
#    setup_maes[setup] = np.mean(np.abs(pred - gt))
#    setup_maxes[setup] = np.max(np.abs(pred - gt))

all_gt = np.hstack(all_gt)

# sorted_maes = sorted(setup_maes.items(), key = lambda kv: kv[1])
# sorted_maxes = sorted(setup_maxes.items(), key = lambda kv: kv[1])

# print(sorted_maes)
# print(sorted_maxes)
# setup_maes_arr = np.array([v for v in setup_maes.values()])
# setup_maxes_arr = np.array([v for v in setup_maxes.values()])
# print("mmae", np.mean(setup_maes_arr))

# print("mean max", np.mean(setup_maxes_arr))
# print("above 5 mae", len(setup_maes_arr[setup_maes_arr>5]))

TP = np.zeros(4)
TN = np.zeros(4)
FP = np.zeros(4)
FN = np.zeros(4)
F1 = np.zeros(4)

sens = np.zeros(4)
spec = np.zeros(4)

plt.show()
plt.close()

strs = ['all', 'valid', 'pref', 'pref_valid', 'other', 'other_valid']

for th_2_class in range(10, 20):
    print("***************", th_2_class, "***************")
    pd_row = []
    pd_row.append(th_2_class)
    pd_labels = []
    for iii, r in enumerate([res]):  # , pref_radar_res, pref_radar_res_v, other_radar_res, other_radar_res_v]):
        # plt.figure()
        fig, ax = plt.subplots(2, 2)
        print(len(r))
        gt4 = []
        gt2 = []

        p4 = []
        p2 = []
        sessions = []
        setups = []

        for k, v in r.items():
            db.update_mysql_db(k)
            sess = db.session_from_setup(k)
            sessions.append(sess)
            setups.append(k)
            mrkr = "^" if k > 100000 else "o"
            th = [5, th_2_class, 30]

            diff = np.round(v[1] - v[0])
            sz = 26
            if np.abs(diff) > 10:
                sz = 36
            st = str(k) + ' ' + str(diff)
            if v[0] < th[0]:
                gt4.append(0)
                gt2.append(0)
                ax[0, 0].scatter(v[0], v[1], c='black', s=sz, marker=mrkr)
                ax[0, 0].scatter(v[0], v[1], c=colors[k % len(colors)], s=24, label=str(k), marker=mrkr)
                ax[0, 0].annotate(st, (v[0], v[1]), fontsize=6)
            elif v[0] < th[1]:
                gt4.append(1)
                gt2.append(0)
                ax[0, 1].scatter(v[0], v[1], c='black', s=sz, marker=mrkr)
                ax[0, 1].scatter(v[0], v[1], c=colors[k % len(colors)], s=24, label=str(k), marker=mrkr)
                ax[0, 1].annotate(st, (v[0], v[1]), fontsize=6)
            elif v[0] < th[2]:
                gt4.append(2)
                gt2.append(1)
                ax[1, 0].scatter(v[0], v[1], c='black', s=sz, marker=mrkr)
                ax[1, 0].scatter(v[0], v[1], c=colors[k % len(colors)], s=24, label=str(k), marker=mrkr)
                ax[1, 0].annotate(st, (v[0], v[1]), fontsize=6)
            else:
                gt4.append(3)
                gt2.append(1)
                ax[1, 1].scatter(v[0], v[1], c='black', s=sz, marker=mrkr)
                ax[1, 1].scatter(v[0], v[1], c=colors[k % len(colors)], s=24, label=str(k), marker=mrkr)
                ax[1, 1].annotate(st, (v[0], v[1]), fontsize=6)
            if v[1] < th[0]:
                p4.append(0)
                p2.append(0)
            elif v[1] < th[1]:
                p4.append(1)
                p2.append(0)
            elif v[1] < th[2]:
                p4.append(2)
                p2.append(1)
            else:
                p4.append(3)
                p2.append(1)

        u = np.unique(sessions)

        cm4 = confusion_matrix(gt4, p4)
        cm2 = confusion_matrix(gt2, p2)

        TP = cm2[1][1]
        TN = cm2[0][0]
        FP = cm2[0][1]
        FN = cm2[1][0]
        # print(TN, FP)
        # print(FN, TP)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        f1 = TP / (TP + 0.5 * (FP + FN))
        print("F1", f1)
        TP = np.zeros(4)
        TN = np.zeros(4)
        FP = np.zeros(4)
        FN = np.zeros(4)
        F1 = np.zeros(4)

        sens = np.zeros(4)
        spec = np.zeros(4)

        # print(i, len(MB_HQ))
        fig.suptitle('Short Train on NWH + MB, Leave 1 Out ' + str(len(r.keys())) + ' setups ' + strs[iii])
        ax[0, 0].set_title('Low. AHI<5' + ' ' + strs[iii])
        ax[0, 1].set_title('Mild. 5<AHI<' + str(th_2_class) + ' ' + strs[iii])
        ax[1, 0].set_title('Moderate. ' + str(th_2_class) + '<AHI<30' + ' ' + strs[iii])
        ax[1, 1].set_title('Severe. AHI>30' + ' ' + strs[iii])

        ax[0, 0].axhline(y=th[0])
        ax[0, 1].axhline(y=th[0])
        ax[0, 1].axhline(y=th[1])
        ax[1, 0].axhline(y=th[2])
        ax[1, 0].axhline(y=th[1])
        ax[1, 1].axhline(y=th[2])
        plt.xlabel('AHI GT')
        plt.ylabel('AHI Pred.')
        # plt.legend(loc='lower right')
        #   plt.show()
        plt.savefig(os.path.join(lp, strs[iii] + '_subplots.png'), dpi=300)
        print("saved", strs[iii] + '.png')
        plt.close()
        for i in range(min(cm4.shape)):
            # print(cm4.shape, TP.shape)
            # print(i)
            TP[i] = cm4[i][i]
            FP[i] = np.sum(cm4[:, i]) - TP[i]
            FN[i] = np.sum(cm4[i, :]) - TP[i]
            TN[i] = np.sum(cm4) - TP[i] - FP[i] - FN[i]
            sens[i] = TP[i] / (TP[i] + FN[i])
            spec[i] = TN[i] / (TN[i] + FP[i])
            F1[i] = 0 if (TP[i] + 0.5 * (FP[i] + FN[i])) == 0 else TP[i] / (TP[i] + 0.5 * (FP[i] + FN[i]))
        print("threshold ", th_2_class, "2 class sensitivity", np.round(sensitivity, 2), "specificity",
              np.round(specificity, 2), "F1", np.round(f1, 2))

        two_class_sens = np.round(sensitivity, 2)
        two_class_spec = np.round(specificity, 2)
        two_class_f1 = np.round(f1, 2)

        pd_row.append(two_class_sens)
        pd_row.append(two_class_spec)
        pd_row.append(two_class_f1)

        for i in range(min(cm4.shape)):
            print("class", i, "sensitivity", sens[i], "specificity", spec[i], "F1", F1[i])

        print(cm4)
        print(cm2)
        sessions = np.unique(sessions)
        print(len(sessions))
        setups = np.array(setups)
        print("NWH Patients", len(sessions[sessions < 100000]))
        print("MB Patients", len(sessions[sessions > 100000]))
        print("NWH Setups", len(setups[setups < 100000]))
        print("MB Setups", len(setups[setups > 100000]))
        print(res_v)
    two_class_outputs_by_threshold.append(pd_row)
ss_df = pd.DataFrame(two_class_outputs_by_threshold,
                     columns=['threshold', '2_class_sens', '2_class_spec', '2_class_f1'])
ss_df.to_csv('/Neteera/Work/homes/dana.shavit/analysis/sens_spec_2class_6100.csv')
