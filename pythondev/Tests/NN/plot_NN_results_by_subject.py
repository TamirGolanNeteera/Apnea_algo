import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import  confusion_matrix
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


lps = ['/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/190323_5sec_huber/',
       '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030423_resnet_mse/',
       '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_weighted_1']
lp_names = ['5s','10s','15s']
lp_colors = ['red', 'green', 'blue']
dirs_dict = {}
ndirs = len(lps)

for lp in lps:
    pred_files = fnmatch.filter(os.listdir(lp), '*_pred.npy')
    label_files = fnmatch.filter(os.listdir(lp), '*_gt.npy')
    dirs_dict[lp] = {}

    for i, fn in enumerate(label_files):
        setup_str = fn[0:fn.find('_')]
        setup = int(setup_str)
        s = db.setup_subject(setup)
        if s not in dirs_dict[lp].keys():
            dirs_dict[lp][s] = []
        dirs_dict[lp][s].append(setup)


for subj in dirs_dict[lps[0]].keys():
    print(subj)
    use = True
    for lp in lps:
        if subj not in dirs_dict[lp].keys():
            use = False
    if not use:
        continue
    fig, ax = plt.subplots(1, 3, figsize=(20,10))
    for lp_i, lp in enumerate(lps):

        for setup in dirs_dict[lp][subj]:#run of setups

            pred_file = os.path.join(lp, str(setup) + '_pred.npy')
            label_file = os.path.join(lp, str(setup) + '_gt.npy')
            pred = np.load(pred_file, allow_pickle=True)
            gt = np.load(label_file, allow_pickle=True)
            pred = pred.ravel()
            preds_avg = rollavg_convolve_edges(pred, 9)

            if len(gt.shape) > 1:
                gt = gt[:, 0]
            gt = gt.ravel()

            mean_gt = np.mean(gt)
            mean_pred = np.mean(preds_avg)

            print(lp, setup, mean_gt, mean_pred, lp[-7:])
            #plt.scatter(mean_gt, np.abs(mean_pred-mean_gt), c='black', s=29,  marker='o', alpha=1)
            #plt.scatter(mean_gt, np.abs(mean_pred-mean_gt), c=colors[setup % len(colors)], s=26, marker='o', alpha=1, label=lp[-7:])
            #plt.text(mean_gt, np.abs(mean_pred-mean_gt), str(setup)+lp[-3:])
            unique_measurements = np.unique(gt)

            mm = np.mean(unique_measurements)
            dist = np.abs(mm - unique_measurements)

            outlier = unique_measurements[np.argmax(dist)]
            mod_X = []
            mod_y = []

            if len(unique_measurements) < 2:
                continue

            unique_gt = []
            unique_pred = []
            for i, m in enumerate(unique_measurements):

                marker = 'o'

                m_gt = gt[gt == m]
                orig_idx = np.where(gt == m)
                m_pred = preds_avg[gt == m]

                m_e = np.mean(np.abs(m_gt - m_pred))

                mean_m_gt = np.mean(m_gt)
                mean_m_pred = np.mean(m_pred)
                unique_gt.append(mean_m_pred)
                unique_pred.append(mean_m_gt)
                ax[lp_i].scatter(mean_m_gt, mean_m_pred, c='black', s=11, marker=marker, alpha=1)
                ax[lp_i].scatter(mean_m_gt, mean_m_pred, c=lp_colors[lp_i], s=8, marker=marker, alpha=1)
                ax[lp_i].text(mean_m_gt, mean_m_pred, str(setup), c=lp_colors[lp_i])
                ax[lp_i].set_title(subj + ' '+ lp_names[lp_i])
                ax[lp_i].set_box_aspect(1)

            if 'stress' in db.setup_note(setup):
                print("STRESS SETUP")
            print(setup, np.round(np.array(unique_pred)))
            print(setup,np.round(np.array(unique_gt)))
    plt.show()
    home = '/Neteera/Work/homes/dana.shavit/'
    fn = subj.replace('_', ' ')+'.png'
    plt.savefig(os.path.join(home, fn))
    plt.close()
