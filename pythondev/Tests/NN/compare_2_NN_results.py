import matplotlib.pyplot as plt

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
import matplotlib.colors as mcolors
col = list(mcolors.cnames.keys())
colors = ['tan', 'lightcoral', 'red', 'green', 'orchid', 'steelblue', 'blue', 'cyan', 'magenta', 'gold', 'darkgreen', 'yellow', 'purple', 'lightblue', 'gray', 'skyblue', 'pink', 'mediumslateblue', 'lightseagreen', 'coral', 'orange', 'violet']


#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210223_large_model/'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150223_no_norm'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150223_no_norm_continue'
lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/220223_back/'
#lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/220223_back_modified'
lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/220223_back_modified_mse'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/230223_front_modified/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210223_no_bn_systolic/'
#lp = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/230223_huge_model/'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/260223_no_norem_mse'
#lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/260223_front_mse_fixed'
#lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/280223_front_huber'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/010323_augment_front'
#lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_weighted_1'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_weighted_mse'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_back'
lp1 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030423_resnet_mse'
#lp2 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/040423_resnet_mse'
lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/090423_resnet_mae'
#lp2 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150423_no_downsample'#030423_resnet_mse'
#lp1 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/170423_128hz'
#lp1 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/170423_nods_mse'#150423_no_downsample'
#lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/150223_no_norm'
#lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/120323_weighted_mse'
lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/090423_resnet_mae'
#lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_with_tokenizr_and_1d'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_mse'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_mae'
#lp1 ='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030423_resnet_mse'
#lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/290423_10sec_no_ds_incr_resnet/'
#lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/030523_transformer_tokenizr_1d_32_mae_ds_reshape_fix'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200423_transformer_tokenizr_1d_32_mae'
#lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_32_mae_ds_reshape_fix/'
#lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_32_mae_ds_reshape_fix_num_tokens8/'


lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_resnet'
lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_resnet_fix_weights'

lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210523_resnet_fix_weights'
#lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_vit'
#lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/300723_transformer_tokenizr_1d_vit_4'
lp1 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/090423_resnet_mae'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/200923_bin_selection/'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/190923_transformer/'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/210523_resnet_fix_weights'
#lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/080523_transformer_tokenizr_1d_32_mae_ds_reshape_fix2/'
#lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/190923_transformer_v2/'
lp1='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/270923_transformer_new2'
# lp2='/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/280923_resnet_with_new_data_creation/'
lp2 = '/Neteera/Work/homes/dana.shavit/Research/BP2023/trained_nns/280923_resnet_with_new_data_creation2/'
print('Processing', lp1)
print('Processing', lp2)

pred_files1 = fnmatch.filter(os.listdir(lp1), '*_pred.npy')
label_files1 = fnmatch.filter(os.listdir(lp1), '*_gt.npy')

pred_files2 = fnmatch.filter(os.listdir(lp2), '*_pred.npy')
label_files2 = fnmatch.filter(os.listdir(lp2), '*_gt.npy')

res = {}

all_gt = []
all_pred = []


perc_5_1 = []
perc_10_1 = []
perc_15_1 = []
perc_out_1 = []
all_setups = []


perc_5_2 = []
perc_10_2 = []
perc_15_2 = []
perc_out_2 = []

total_loss_highs1 = []
total_loss_highs2 = []

total_loss_lows1 = []
total_loss_lows2 = []

total_loss_mid1 = []
total_loss_mid2 = []
all_setups_measures = []
completed_subjects = []
sum_err1 = 0
sum_err2 = 0
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
names1 = []
names2 = []
setups1 = []
setups2 = []

err1 = []
err2 = []

for i, fn in enumerate(label_files1):
    setup_str = fn[0:fn.find('_')]
    setup = int(setup_str)
    sname = db.setup_subject(setup)
    names1.append(sname)
    setups1.append(setup)
for i, fn in enumerate(label_files2):
    setup_str = fn[0:fn.find('_')]
    setup = int(setup_str)
    sname = db.setup_subject(setup)
    names2.append(sname)
    setups2.append(setup)


names1 = np.unique(np.stack(names1))
names2 = np.unique(np.stack(names2))
names = list(set(names1).intersection(set(names2)))

setups1 = np.unique(np.stack(setups1))
setups2 = np.unique(np.stack(setups2))
setups = list(set(setups1).intersection(set(setups2)))

print(names)
print(len(names))

totalloss1 = 0
totalloss2 = 0
counter1 = 0
counter2 = 0

setups_by_bp = {100: [], 115: [], 125: [], 135: [], 145: [], 155: [], 165: []}#{100: [], 110: [], 120: [], 130: [], 140: [], 150: [], 160: [], 170: []}
class_gt = []
class_pred1 = []
class_pred2 = []

bp_levels = [140, 130, 120, 110]

def bp_to_class(bp):
    if bp > bp_levels[0]:
        return 4
    elif bp > bp_levels[1]:
        return 3
    elif bp > bp_levels[2]:
        return 2
    elif bp > bp_levels[3]:
        return 1
    else:
        return 0

for i, setup in enumerate(setups):

    sname = db.setup_subject(setup)

    b = [ord(c) for c in sname]
    d = db.setup_distance(setup)
    # if d != 1000:
    #      continue
    b = np.where(np.array(names) == sname)[0][0]
    #print("------------", sname, setup, d)
    gt1 =  np.load(os.path.join(lp1, str(setup) + '_gt.npy'), allow_pickle=True)
    pred1 = np.load(os.path.join(lp1, str(setup) + '_pred.npy'), allow_pickle=True)
    pred1 = pred1.ravel()
    preds_avg1 = rollavg_convolve_edges(pred1, 9)

    if len(gt1.shape) == 2:
        gt1 = gt1[:, 0]
    st = 0#int(len(gt)*0.2)
    mean_gt1 = np.mean(gt1[st:])

    mean_pred1 = np.mean(preds_avg1[st:])

    gt2 = np.load(os.path.join(lp2, str(setup) + '_gt.npy'), allow_pickle=True)
    pred2 = np.load(os.path.join(lp2, str(setup) + '_pred.npy'), allow_pickle=True)
    pred2 = pred2.ravel()
    preds_avg2 = rollavg_convolve_edges(pred2, 9)

    if len(gt2.shape) == 2:
        gt2 = gt2[:, 0]

    mean_gt2 = np.mean(gt2)
    mean_pred2 = np.mean(preds_avg2[st:])
    if mean_pred1 < 60 or mean_pred2 < 60:
        continue
    class_gt.append(bp_to_class(mean_gt1))
    class_pred1.append(bp_to_class(mean_pred1))
    class_pred2.append(bp_to_class(mean_pred2))

    thresh_5 = 5
    thresh_10 = 10
    thresh_12 = 12.5
    thresh_15 = 15

    if gt1.shape != preds_avg1.shape:
        continue
    if gt2.shape != preds_avg2.shape:
        continue
    loss1 = np.abs(mean_gt1-mean_pred1)#np.mean(np.abs(gt1-preds_avg1))
    loss2 = np.abs(mean_gt2-mean_pred2)#np.mean(np.abs(gt2-preds_avg2))

    sum_err2 += loss2
    sum_err1 += loss1

    dict_keys = np.sort(list(setups_by_bp.keys()))[::-1]

    placed  = False
    for k in dict_keys:
        if mean_gt1 >= k:
            setups_by_bp[k].append([setup, loss1, loss2, mean_gt1])
            placed = True
            break
    if not placed:
        setups_by_bp[100].append([setup, loss1, loss2, mean_gt1])


    if mean_gt1 > 150:
        total_loss_highs1.append(loss1)
        total_loss_highs2.append(loss2)
    elif mean_gt1 < 110:
        total_loss_lows1.append(loss1)
        total_loss_lows2.append(loss2)
    else:

        total_loss_mid1.append(loss1)
        total_loss_mid2.append(loss2)

    if loss1 < thresh_5:
        perc_5_1.append(setup)
    elif loss1 < thresh_10:
        perc_10_1.append(setup)
    elif loss1 < thresh_15:
        perc_15_1.append(setup)
    else:
        perc_out_1.append(setup)



    if loss2 < thresh_5:
        perc_5_2.append(setup)
    elif loss2 < thresh_10:
        perc_10_2.append(setup)
    elif loss2 < thresh_15:
        perc_15_2.append(setup)
    else:
        perc_out_2.append(setup)


    all_setups.append(setup)
    
    ax[0].scatter(mean_gt1, mean_pred1, c='black', s=24,  marker='o', alpha=1)
    ax[1].scatter(mean_gt2, mean_pred2, c='black', s=24,  marker='^', alpha=1)
    if db.setup_subject(setup) in completed_subjects:
        ax[0].scatter(mean_gt1, mean_pred1, c=colors[b % len(colors)], s=21, marker='o', alpha=1)
        ax[1].scatter(mean_gt2, mean_pred2, c=colors[b % len(colors)], s=21, marker='^', alpha=1)
        #ax[0].plot([mean_gt1, mean_gt1], [mean_pred1, mean_pred2], '-k', linewidth=0.5)
        #ax[1].plot([mean_gt1, mean_gt1], [mean_pred1, mean_pred2], '-k', linewidth=0.5)
    else:
        ax[0].scatter(mean_gt1, mean_pred1, c=colors[b % len(colors)], s=21, marker='o', alpha=1, label=sname+" 1")
        ax[1].scatter(mean_gt2, mean_pred2, c=colors[b % len(colors)], s=21, marker='^', alpha=1,label=sname+" 2")
        #ax[0].plot([mean_gt1, mean_gt1], [mean_pred1, mean_pred2], '-k', linewidth=0.5)
        #ax[1].plot([mean_gt1, mean_gt1], [mean_pred1, mean_pred2], '-k', linewidth=0.5)


    #plt.text(mean_gt, mean_pred, str(d))
    for f in [ax[0], ax[1]]:
        f.plot([80,200], [80,200], '--k', linewidth=0.5)
        f.plot([80,190], [90,200], '--k', linewidth=0.5)
        f.plot([90,200], [80,190], '--k', linewidth=0.5)


    unique_measurements = np.unique(gt1)

    for i,m in enumerate(unique_measurements):
    #     if i == 0:
    #         pcolor = colors[np.sum(b) % len(colors)]
    #         marker = 'x'
    #     else:
    #         pcolor = colors[np.sum(b) % len(colors)]
    #         marker='o'
        m_gt1 = gt1[gt1 == m]
        m_gt2 = gt2[gt2 == m]
        m_pred1 = preds_avg1[gt1 == m]
        m_pred2 = preds_avg2[gt2 == m]
        m_loss1 = np.mean(np.abs(m_gt1 - m_pred1))
        m_loss2 = np.mean(np.abs(m_gt2 - m_pred2))

        if m_loss2 - m_loss1 > 5:#m_loss1 < m_loss2:
            counter1+=1
        elif m_loss1 - m_loss2 > 5:
            counter2+=1
        totalloss1 += loss1
        totalloss2 += loss2


    #     m_e = np.mean(np.abs(m_gt - m_pred))
    #     mean_m_gt = np.mean(m_gt)
    #     mean_m_pred = np.mean(m_pred)
    #     thresh_5 = 0.05*mean_m_gt
    #     thresh_10 = 0.1*mean_m_gt
    #
    #     if np.abs(mean_m_pred-mean_m_gt) < thresh_5:
    #         perc_5_measures.append(setup)
    #     elif np.abs(mean_m_pred-mean_m_gt) < thresh_10:
    #         perc_10_measures.append(setup)
    #     else:
    #         perc_out_measures.append(setup)
    #     all_setups_measures.append(setup)
    #     all_gt.append(mean_m_gt)
    #     all_pred.append(mean_m_pred)
    #     plt.scatter(mean_m_gt, mean_m_pred, c='black', s=11, marker=marker, alpha=1)
    #     plt.scatter(mean_m_gt, mean_m_pred, c=colors[(len(b)*np.sum(b)) % len(colors)], s=8, marker=marker, alpha=1)

    completed_subjects.append(sname)
    db.update_mysql_db(setup)
print("counter1", counter1, "counter2", counter2)
print("totalloss1", totalloss1, "totalloss2", totalloss2)
# print("measurements",len(perc_5_measures)/len(all_setups_measures), "<5%", len(perc_10_measures)/len(all_setups_measures), "<10%", len(perc_out_measures)/len(all_setups_measures), ">10%")
print(lp1)
print("setups",100*len(perc_5_1)/len(setups), "<5%", 100*len(perc_10_1)/len(setups), "<10%", 100*len(perc_15_1)/len(setups), "<15%", 100*len(perc_out_1)/len(setups), ">15%")
print(lp2)
print("setups",100*len(perc_5_2)/len(setups), "<5%", 100*len(perc_10_2)/len(setups), "<10%", 100*len(perc_15_2)/len(setups), "<15%", 100*len(perc_out_2)/len(setups), ">15%")
print(100*len(perc_5_1)/len(setups) + 100*len(perc_10_1)/len(setups)," < 10mmHg", 100*len(perc_15_1)/len(setups)+100*len(perc_out_1)/len(setups) ,"> 10 mmHg")
print(100*len(perc_5_2)/len(setups) + 100*len(perc_10_2)/len(setups)," < 10mmHg", 100*len(perc_15_2)/len(setups)+100*len(perc_out_2)/len(setups) ,"> 10 mmHg")
for f in range(len(ax)):
    ax[f].set_title("BP Predictions Systolic, Front, 10 sec. \n1:" +  lp1[lp1.rfind('/'):]+', ERR: '+str(np.round(sum_err1/len(setups),2))+' \n2:'+ lp2[lp2.rfind('/'):]+', ERR: '+str(np.round(sum_err2/len(setups),2)))
#ax[1].legend(loc='upper right')

total_loss_highs1 = np.sum(total_loss_highs1)/len(total_loss_highs1)
total_loss_highs2= np.sum(total_loss_highs2)/len(total_loss_highs2)

total_loss_lows1= np.sum(total_loss_lows1)/len(total_loss_lows1)
total_loss_lows2= np.sum(total_loss_lows2)/len(total_loss_lows2)

total_loss_mid1 = np.sum(total_loss_mid1)/len(total_loss_mid1)
total_loss_mid2 = np.sum(total_loss_mid2)/len(total_loss_mid2)


print("loss low:", total_loss_lows1, total_loss_lows2)
print("loss mid:", total_loss_mid1, total_loss_mid2)
print("loss high:", total_loss_highs1, total_loss_highs2)
plt.show()
for k,v in setups_by_bp.items():
    print(k, len(v), "setups")
    s1 = 0
    s2 = 0

    for vv in v:
        s1+=vv[1]
        s2+=vv[2]
    s1/=len(v)
    s2/=len(v)
    print(s1, s2)
fn = 'bp_predictions_systolic.png'
plt.savefig(os.path.join(lp1, fn))
plt.close()

class_gt = np.stack(class_gt)
class_pred1 = np.stack(class_pred1)
class_pred2 = np.stack(class_pred2)
cm1 = confusion_matrix(class_gt, class_pred1)
cm2 = confusion_matrix(class_gt, class_pred2)
print(cm1)
print(cm2)