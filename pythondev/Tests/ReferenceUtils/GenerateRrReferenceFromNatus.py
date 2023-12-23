from scipy.signal import find_peaks

from VSProcessor import autocorr_measurement

from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import DB, Sensor, VS

from pylibneteera.float_indexed_array import TimeArray

import numpy as np
import pandas as pd

LENGTH = 30

sess = 6633


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


if __name__ == '__main__':
    db = DB()
    for sess in {db.setup_multi(x)[0] for x in db.setup_by_project('nwh')}:
        print(sess)
        chest = load_reference(sess, 'chest', db)
        if chest is None:
            continue
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

        df_copy = df.copy()
        df[f'chest_pred'][df[f'chest_qual'] < 0.6] = np.nan
        df[f'chest_pred'][df[f'chest_pred'] == -1] = np.nan
        df[f'chest_qual'][df[f'chest_qual'] == 0] = np.nan
        path = db.setup_ref_path_npy(sess, Sensor.natus, VS.rr)
        np.save(path, df.chest_pred.values.astype(float))
