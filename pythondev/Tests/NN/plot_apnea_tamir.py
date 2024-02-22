import fnmatch
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from Tests.NN.create_apnea_count_AHI_data import compute_phase, getSetupRespiration, qnormalize
from Tests.NN.predict_apnea_on_cloud_setup_data import device_map
from Tests.vsms_db_api import DB
from Tests.NN.tamir_setups_stitcher import stitch_and_align_setups_of_session

base_path = '/Neteera/Work/homes/tamir.golan/Apnea_data/embedded_Model_2/scaled/'
compute_phase()
resp = getSetupRespiration(6284)
setups = [6284]
db = DB()
db.update_mysql_db(setups[0])
for i_sess, sess in enumerate(sessions):
    phase_dir = None
    phase_dir = '/Neteera/Work/homes/tamir.golan/embedded_phase/MB_old_and_new_model_2_311223/hr_rr_ra_ie_stat_intra_breath_31_12_2023/accumulated'
    fs = 10 if phase_dir else 500
    try:
        stitch_dict = stitch_and_align_setups_of_session(sess, phase_dir=phase_dir)
    except:
        continue
    for setups in list(stitch_dict):
        print(f'process {setups}')
        stat_data = stitch_dict[setups]['stat']

        ss_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*sleep*'))[0], allow_pickle=True)
        apnea_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*pnea*'))[0], allow_pickle=True)
        sp02 = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*SpO2*'))[0], allow_pickle=True)[::50]
        print(sess, ' apneas ', str(len(ss_ref) > len(apnea_ref)))
        hr_ref = np.load(list(Path(db.setup_dir(setups[0])).parent.rglob('*HR*'))[0], allow_pickle=True)
        if stitch_dict[setups]['ref_earlier']:
            apnea_ref = apnea_ref[stitch_dict[setups]['gap'] // fs:]
            sp02 = sp02[stitch_dict[setups]['gap'] // fs:]
        setup_files = fnmatch.filter(os.listdir(base_path),'*_X.npy')
        label_files = fnmatch.filter(os.listdir(base_path),'*_y.npy')
        valid_files = fnmatch.filter(os.listdir(base_path),'*_valid.npy')
        ss_files = fnmatch.filter(os.listdir(base_path),'*_ss_ref*.npy')
        empty_files = fnmatch.filter(os.listdir(base_path),'*_empty*.npy')
        apnea_files = fnmatch.filter(os.listdir(base_path),'*_apnea*.npy')
        for setup in db.setups_by_session(sess):
            try:
                X_fn = [f for f in setup_files if str(setup) in f][0]
                y_fn = [f for f in label_files if str(setup) in f][0]
                v_fn = [f for f in valid_files if str(setup) in f][0]
                a_fn = [f for f in apnea_files if str(setup) in f][0]
                s_fn = [f for f in ss_files if str(setup) in f][0]
                X = np.load(os.path.join(base_path, X_fn), allow_pickle=True)
                y = np.load(os.path.join(base_path, y_fn), allow_pickle=True)
                valid = np.load(os.path.join(base_path, v_fn), allow_pickle=True)
                ss = np.load(os.path.join(base_path, s_fn), allow_pickle=True)
                apnea = np.load(os.path.join(base_path, a_fn), allow_pickle=True)
                device_id = db.setup_sn(setup)[0]
                device_location = device_map[int(device_id) % 1000]
                if device_location == 1:
                    pass
            except:
                continue

