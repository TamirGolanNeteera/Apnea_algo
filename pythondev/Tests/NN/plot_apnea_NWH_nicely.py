import fnmatch
import os
import pandas as pd
from pathlib import Path
from scipy import signal as sp

import numpy as np
import matplotlib.pyplot as plt
from Tests.NN.create_apnea_count_AHI_data import getSetupRespiration, qnormalize, compute_respiration
from Tests.NN.predict_apnea_on_cloud_setup_data import device_map
from Tests.vsms_db_api import DB, Sensor, VS
from Tests.Utils.LoadingAPI import load_reference
from Tests.NN.tamir_setups_stitcher import stitch_and_align_setups_of_session

apnea_class = {'missing': -1,
        'Normal Breathing': 0,
        'normal': 0,
               'Apnea' : 1,
        'Central Apnea': 1,
        'Hypopnea': 2,
        'Mixed Apnea': 3,
        'Obstructive Apnea': 4,
'Obstructive Hypopnea' : 6,
        'Noise': 5}

setup = 113390
session = 109257
db = DB()
db.update_mysql_db(setup)
# apnea_ref = load_reference(setup, 'apnea', db)
if setup < 10000:
        resp = getSetupRespiration(setup)[0]
else:

        stitch_dict = stitch_and_align_setups_of_session(db.session_from_setup(setup))
        phase = list(stitch_dict.values())[0]['phase']
        phase_df = pd.DataFrame(phase)
        phase_df.interpolate(method="linear", inplace=True)
        phase_df.fillna(method="bfill", inplace=True)
        phase_df.fillna(method="pad", inplace=True)
        phase = phase_df.to_numpy().flatten()
        respiration = compute_respiration(phase, lp=0.05, hp=3.33)
        resp = sp.resample_poly(respiration, 1, 50)
        gap_1_hz = list(stitch_dict.values())[0]['gap'] // 500
        apnea_ref = apnea_ref[gap_1_hz:]
plt.plot(resp, linewidth=0.5)
apnea_ref = np.repeat([apnea_class[ap] if ap is not None else 0 for ap in apnea_ref], 10)
plt.plot(apnea_ref, linewidth=0.5)
pass