import fnmatch
import os
from pathlib import Path

import numpy as np
from Tests.NN.create_apnea_count_AHI_data import NW_HQ, apnea_class
from Tests.vsms_db_api import DB
from Tests.Utils.LoadingAPI import load_reference



db = DB()
AHI_dict = {}
for setup in NW_HQ:
    apnea_ref = load_reference(setup, 'apnea', db)
    apnea_ref = [apnea_class[ap] if ap is not None else 0 for ap in apnea_ref]
    apnea_ref[apnea_ref == -1] = 0
    apnea_diff = np.diff(apnea_ref, prepend=0)
    apnea_events = len(apnea_diff[apnea_diff>0])
    ss_ref = load_reference(setup, 'sleep_stages', db)
    ss_ref = np.array([0 if ss == 'W' else 1 for ss in ss_ref])
    sleep_sec = np.count_nonzero(ss_ref)
    AHI = apnea_events / (sleep_sec / 3600)
    AHI_dict[setup] = AHI
    print(AHI)
print(AHI_dict)