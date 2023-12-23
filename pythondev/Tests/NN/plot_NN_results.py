import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
import  fnmatch
from Tests.vsms_db_api import DB
import numpy as np
import os
db = DB()

import scipy as sci

def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')

col = list(mcolors.cnames.keys())
colors = ['tan', 'lightcoral', 'red', 'green', 'blue', 'cyan', 'magenta', 'gold', 'darkgreen', 'yellow', 'purple', 'lightblue', 'gray', 'skyblue', 'pink', 'mediumslateblue', 'lightseagreen', 'coral', 'orange', 'violet']


lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210223_large_model/'


#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/220223_back/'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/220223_back_modified/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/230223_front_modified/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210223_no_bn_systolic/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/230223_huge_model/'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/260223_no_norem_mse/'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/260223_front_mse_fixed'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/010323_augment_front'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/070323_skip'

lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080323_weighted'#150223_no_norm'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_back'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150223_no_norm'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/190323_5sec_huber/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200323_15sec/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_weighted_1'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030423_resnet_mse/'
# lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150423_no_downsample/'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_with_tokenizr_and_1d'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_resnet_15sec_nods/'#/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_15s_nods_mae'#200423_transformer_tokenizr_1d_32_mae'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_mae'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/290423_10sec_no_ds_incr_resnet/'#290423_10sec_no_ds/'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/290423_10sec_no_ds_resnet'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_15s_nods_mae'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/0205_transformer_/'
print('Processing', lp)
# 200423_transformer_with_tokenizr_and_1d_32
# 200423_transformer_tokenizr_1d_32_mse_reducelr
# 200423_transformer_tokenizr_1d_32
# 200423_transformer_tokenizr_1d_32_mae
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_15s_nods_mae'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/0305_transformer_2'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030523_transformer_tokenizr_1d_15sec'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_mae'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_32_mae_ds_reshape_fix2/'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_resnet'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_resnet_fix_weights'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210523_resnet_fix_weights'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_vit'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/300723_transformer_tokenizr_1d_vit_4'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/290423_10sec_no_ds_resnet'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030423_resnet_mse/'
#lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200923_bin_selection/'
lp='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/270923_transformer_new2'
lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/280923_resnet_with_new_data_creation2/'
pred_files = fnmatch.filter(os.listdir(lp), '*_pred.npy')
label_files = fnmatch.filter(os.listdir(lp), '*_gt.npy')
#data_files =  fnmatch.filter(os.listdir(lp), '*_X.npy')

res = {}

all_gt = []
all_pred = []

y_true = []
y_pred = []

p_ok = []
p_not_ok = []
perc_5 = []
perc_10 = []
perc_15 = []
perc_out = []
all_setups = []

loss_by_dist = {500:[], 1000:[], 1500:[]}

perc_5_measures = []
perc_10_measures = []
perc_15_measures = []
perc_out_measures = []
all_setups_measures = []
completed_subjects = []
sum_err = 0
plt.figure(figsize=(10,10))
names = []#['Vlad Lirtsman','Rakefet Shohat', 'David Grossman', 'Ohad Basha', 'Ariel Segal', 'P15']#, 'Amichy Feldman' ,'Hilat Doron ' ,'Cie Mintz', 'Ehud Fishler', 'Arkady Pann']
#
n = ['Amichy Feldman', 'Cie Mintz']
#  #    , 'Ehud Fishler', 'Anton Shor', 'Ariel Segal', 'Arkady Pann', 'Cie Mintz',  'Dana Shavit', 'David Dadoune', 'David Grossman', 'Dean Ranmar',
#  # 'Eldar Hacohen', 'Hilat Doron ', 'Idan Yona', 'Isaac Litman',  'Itai Efrat', 'Itamar Saban', 'Linor Osipov', 'Lior Oron', 'Lital Alon',
#  # 'Melanie Grably', 'Michael Hirsh', 'Moshe Aboud', 'Moshe Caspi',  'Nachum Shtauber', 'Nati Ben-Mordehai', 'Nati Edri', 'Nativ Zohar',
#  # 'Neomi Pann', 'Noa Simon', 'Ohad Basha', 'P11', 'P12', 'P13', 'P14', 'P16', 'P17',  'P19', 'P2', 'P4', 'P5', 'P6', 'P8', 'Rael Cohen', 'Rakefet Shohat',
#  # 'Rani Shifron', 'Reinier Doelman', 'Rephael Grably', 'Shahar Yaron',  'Vered Fainsod', 'Vlad Lirtsman', 'Yoni Levi', 'aviv g', 'linda', 'orit']
counted_setups = 0
for i, fn in enumerate(label_files):

    setup_str = fn[0:fn.find('_')]
    setup = int(setup_str)
    b = [ord(c) for c in db.setup_subject(setup)]

    d = db.setup_distance(setup)

    print(db.setup_subject_details(setup))
    # if d != 1000:
    #     continue
    print("------------", db.setup_subject(setup), setup, d)
    gt = np.load(os.path.join(lp, fn), allow_pickle=True)
    pred = np.load(os.path.join(lp, setup_str + '_pred.npy'), allow_pickle=True)
    #X = np.load(os.path.join(lp, setup_str + '_X.npy'), allow_pickle=True)
    pred = pred.ravel()

    preds_avg = rollavg_convolve_edges(pred, 9)

    if len(gt.shape) > 1:
        gt = gt[:,0]
    gt = gt.ravel()




    #
    # if (gt<80).any():
    #     gt[gt < 80] = np.mean(gt[gt>=80])
    #     print("***", setup)
    # if (gt>210).any():
    #     gt[gt > 210] = np.mean(gt[gt <= 210])
    #     print("***", setup)

    mean_gt = np.mean(gt)
    mean_pred = np.mean(preds_avg)

    if mean_pred < 60:
        continue
    # if db.setup_subject(setup) not in n:
    #     continue

    sum_err += np.abs(mean_pred-mean_gt)

    counted_setups += 1
    names.append(db.setup_subject(setup))
    thresh_5 = 5
    thresh_10 = 10
    thresh_15 = 15

    if gt.shape != preds_avg.shape:
        continue

    #loss = np.mean(np.abs(gt-preds_avg))
    loss = np.abs(mean_gt-mean_pred)
    if d not in loss_by_dist.keys():
        loss_by_dist[d] = []
    loss_by_dist[d].append(loss)
    if mean_gt > 140 and mean_pred > 140:
        p_ok.append(setup)
    elif mean_gt < 115 and mean_pred < 115:
        p_ok.append(setup)
    elif mean_gt >= 115 and mean_pred >= 115 and  mean_gt <= 140 and mean_pred <= 140:
        p_ok.append(setup)
    else:
        p_not_ok.append(setup)
        
    if loss < thresh_5:
        perc_5.append(setup)

    elif loss < thresh_10:
        perc_10.append(setup)

    elif loss < thresh_15:
        perc_15.append(setup)
    else:
        perc_out.append(setup)
    all_setups.append(setup)

    alpha = 1
    if d != 1000:
        alpha = 1

    unique_measurements = np.unique(gt)
    print(setup, unique_measurements)

    mm = np.mean(unique_measurements)
    dist = np.abs(mm-unique_measurements)
    pred_dist = []
    outlier = unique_measurements[np.argmax(dist)]

    for i,m in enumerate(unique_measurements):

        #if len(unique_measurements) > 2:
        #    if m == outlier:
        #        continue

        # if i == 0:
        #     pcolor = colors[np.sum(b) % len(colors)]
        #     marker = 'x'
        # else:
        #     pcolor = colors[np.sum(b) % len(colors)]
        #     marker='o'

        marker='o'
        pcolor = colors[np.sum(b) % len(colors)]
        m_gt = gt[gt == m]
        orig_idx = np.where(gt==m)
        m_pred = preds_avg[gt == m]
        m_pred_i = [int(ppp) for ppp in m_pred]
        print("gt", m_gt[0], m_pred_i)
        m_e = np.mean(np.abs(m_gt - m_pred))

        mean_m_gt = np.mean(m_gt)

        mean_m_pred = np.mean(m_pred)
        pred_dist.append(mean_m_pred)
        thresh_5 = 5
        thresh_10 = 10
        thresh_15 = 15

        #loss = np.mean(np.abs(m_gt - m_pred))
        #print(loss)
        loss = np.abs(mean_m_gt - mean_m_pred)
        #print(loss)
        if 'stress' in db.setup_note(setup):
            print("STRESS SETUP")
        print(setup, "gt", mean_m_gt, "p", mean_m_pred, "loss",loss, "dist from mean", dist[i])
        if loss < thresh_5:
            perc_5_measures.append(setup)
        elif loss < thresh_10:
            perc_10_measures.append(setup)
        elif loss < thresh_15:
            perc_15_measures.append(setup)
        else:
            perc_out_measures.append(setup)
        all_setups_measures.append(setup)
        all_gt.append(mean_m_gt)
        all_pred.append(mean_m_pred)
        #plt.scatter(mean_m_gt, mean_m_pred, c='black', s=11, marker=marker, alpha=1)
        #plt.scatter(mean_m_gt, mean_m_pred, c=colors[(len(b)*np.sum(b)) % len(colors)], s=8, marker=marker, alpha=1)
        #plt.text(mean_m_gt, mean_m_pred, str(setup), fontsize=5, alpha=0.5)

        if mean_m_gt > 140:
            y_true.append(2)
        elif mean_m_gt < 115:
            y_true.append(0)
        else:
            y_true.append(1)
        if mean_m_pred > 140:
            y_pred.append(2)
        elif mean_m_pred < 115:
            y_pred.append(0)
        else:
            y_pred.append(1)

    print(pred_dist)
    measurement_pred_mean = np.mean(pred_dist)
    dist = np.abs(measurement_pred_mean-pred_dist)
    outlier = pred_dist[np.argmax(dist)]
    print(outlier)
    for im, m in enumerate(pred_dist):
        if m == outlier:
            alpha = 0.1
        else:
            alpha=1
        plt.scatter(unique_measurements[im], m, c='black', s=11, marker=marker, alpha=alpha)
        plt.scatter(unique_measurements[im], m, c=colors[(len(b)*np.sum(b)) % len(colors)], s=8, marker=marker, alpha=alpha)
    # plt.text(mean_m_gt, mean_m_pred, str(setup), fontsize=5, alpha=0.5)
    #plt.scatter(mean_gt, mean_pred, c='black', s=29, marker='^', alpha=alpha)

    # if db.setup_subject(setup) in completed_subjects:
    #     plt.scatter(mean_gt, mean_pred, c=colors[(len(b) * np.sum(b)) % len(colors)], s=26, marker='^', alpha=alpha)
    # else:
    #     plt.scatter(mean_gt, mean_pred, c=colors[(len(b) * np.sum(b)) % len(colors)], s=26, marker='^', alpha=alpha,
    #                 label=db.setup_subject(setup))
    # plt.text(mean_gt, mean_pred, db.setup_subject(setup), fontsize=4, alpha=alpha)
    plt.plot([80, 200], [80, 200], '--k', linewidth=0.5)
    plt.plot([80, 190], [90, 200], '--k', linewidth=0.5)
    plt.plot([90, 200], [80, 190], '--k', linewidth=0.5)
    plt.title(lp)

    print("completed setup", setup)
    completed_subjects.append(db.setup_subject(setup))
    db.update_mysql_db(setup)

names = np.unique(np.stack(names))
print(len(names), names)
cm=confusion_matrix(np.hstack(y_true), np.hstack(y_pred))
print()

print("measurements",100*len(perc_5_measures)/len(all_setups_measures), "<5%",100*len(perc_10_measures)/len(all_setups_measures), "<10%",100*len(perc_15_measures)/len(all_setups_measures), "<15%", 100*len(perc_out_measures)/len(all_setups_measures), ">15%")
print("setups",100*len(perc_5)/len(all_setups), "<5%", 100*len(perc_10)/len(all_setups), "<10%", 100*len(perc_15)/len(all_setups), "<15%",100*len(perc_out)/len(all_setups), ">15%")
print("setups binary",100*len(p_ok)/len(all_setups), "ok", 100*len(p_not_ok)/len(all_setups), "not ok")
plt.title("BP Predictions Systolic, Front, 10 sec. " + str(np.round(sum_err/counted_setups,2)))
#plt.legend(loc='lower left')

for k,v in loss_by_dist.items():
    print(k, np.mean(v))

plt.show()
home = '/Neteera/Work/homes/dana.shavit/'
fn = 'bp_predictions_systolic_15sec.png'
plt.savefig(os.path.join(home, fn))
plt.close()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
fn = 'bp_predictions_systolic_15sec_cm.png'
plt.savefig(os.path.join(home, fn))
plt.show()
plt.close()

