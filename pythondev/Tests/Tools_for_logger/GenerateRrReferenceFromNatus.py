
from VSProcessor import autocorr_measurement

from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import DB, Sensor, VS
from Tests.Utils.ResearchUtils import print_var

from pylibneteera.float_indexed_array import TimeArray

import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks

LENGTH = 30


def get_rr(sig):
    pred, qual, val = autocorr_measurement(sig, 0.25, 8)
    disp_normed = sig - np.min(sig)
    disp_normed /= np.max(disp_normed)
    peaks = find_peaks(disp_normed, height=0.8)[0]
    if pred > 45 or pred < 4 or len(peaks) == 1:
        qual = 0
    elif pred > 20:
        pred_1, qual_1, val_1 = autocorr_measurement(sig[:int(len(sig) / 2)], 0.25, 8)
        pred_2, qual_2, val_2 = autocorr_measurement(sig[int(len(sig) / 2):], 0.25, 8)
        qual = min(qual_1 * val_1, qual_2 * val_2) * (abs(pred_2 - pred_1) < 3)
    qual /= (1 - sig.fs * 30 / pred / len(sig))  # normalize convolution length
    return pred, qual


def calc_rr_from_ref(setup, db):
    print(setup)
    rr_ref = load_reference(setup, 'rr', db)
    print_var(len(rr_ref) / 3600)
    if len(rr_ref) / 3600 < 15:
        return
    chest = load_reference(setup, 'chest', db)
    if chest is None:
        print_var(chest)
        return
    chest = TimeArray(chest, 1/512)

    df = pd.DataFrame(columns=['chest_pred', 'chest_qual'])
    for i in range(0, int(len(chest) / chest.fs) - LENGTH):
        if i < LENGTH:
            df.loc[i, :] = [-1, 0]
        else:
            displacement = chest[i * chest.fs: (i + LENGTH) * chest.fs]
            max_min_diff = np.max(displacement) - np.min(displacement)
            if max_min_diff > 1:
                pred_chest = -1
                qual_chest = 0
            else:
                disp_pre = displacement[::32].detrend()
                pred_chest, qual_chest = get_rr(disp_pre)
            df.loc[i, :] = [pred_chest, qual_chest]

    df[f'chest_pred'][df[f'chest_qual'] < 0.6] = np.nan
    df[f'chest_pred'][df[f'chest_pred'] == -1] = np.nan
    df[f'chest_qual'][df[f'chest_qual'] == 0] = np.nan
    path = db.setup_ref_path_npy(setup, Sensor.natus, VS.rr)
    if os.path.exists(path):
        os.remove(path)
    np.save(path, df.chest_pred.values.astype(float))


if __name__ == '__main__':
    db = DB()
    calc_rr_from_ref(9031, db)
    exit()
    for setup in {db.setup_multi(x)[0] for x in db.benchmark_setups('nwh')}:
        calc_rr_from_ref(setup, db)

