from Tests.Utils.LoadingAPI import load_reference
from Tests.Utils.PandasUtils import pd_str_plot
from Tests.Utils.PathUtils import create_dir
from Tests.Utils.ResearchUtils import print_var
from Tests.Utils.TestsUtils import is_debug_mode
from Tests.vsms_db_api import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from pylibneteera.ExtractPhaseFromCPXAPI import get_fs

class_to_label = {0: 'normal', 1: 'hypopnea', 2: 'central apnea', 3: 'obstructive apnea', 4: 'mixed apnea', 5: 'apnea',
                  np.nan: 'none'}
label_to_class = {v: k for k, v in class_to_label.items()}
event = 'Hypopnea'
FOLDER = f'/Neteera/Work/homes/moshe.caspi/projects/apnea/{event.split()[0].lower()}_ref_validation'


def plot_hypopnea_event(setup, signal, index, note, show=True):
    signal_to_plot = signal[::50].copy()
    pd_str_plot(signal_to_plot)
    plt.xlabel('Time [sec]')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(FOLDER, f'{setup}_{event.split()[0].lower()}_event_sec_{index}.png'))
        plt.close()


def event_starting_points(ref_apnea, event_types: list):
    ref_apnea = np.array(ref_apnea)
    events = []
    is_event_in_progress = False
    for i, ref_sec in enumerate(ref_apnea):
        if isinstance(ref_sec, str) and ref_sec in event_types and not is_event_in_progress:
            events.append(i)
        is_event_in_progress = isinstance(ref_sec, str) and ref_sec in event_types
    return events


def plot_single_setup(idx, db):
    reference = load_reference(idx, ['chest', 'nasalpress', 'apnea', 'spo2'], db).fillna(method='ffill')
    fs_ref = 512#get_fs(reference)
    reference.loc[reference[reference.spo2 < 1].index, 'spo2'] = np.nan

    events_indices = event_starting_points(reference.apnea, event)
    note = db.setup_note(idx)
    for hyp_start in events_indices:
        print_var(hyp_start)
        raw_window = reference.iloc[hyp_start - 60 * fs_ref: hyp_start + 60 * fs_ref]
        plot_hypopnea_event(idx, raw_window, hyp_start, note)


if __name__ == '__main__':

    from Tests.vsms_db_api import *

    db = DB('neteera_cloud_mirror')

    sessions   = [106413, 106419, 106422, 106424]
    for sess in sessions:
        print(sess)
        db.update_mysql_db(sess)
        setups = db.setups_by_session(sess)
        #   db.session_from_setup(setups[0])

        print(setups)
        for s in setups:
            sn = db.setup_sn(s)
            print(s, sn)





    db = DB()
    create_dir(FOLDER)
    if not is_debug_mode():
        matplotlib.use('agg')
    for setup_id in db.benchmark_setups(Benchmark.nwh):
        print_var(setup_id)
        plot_single_setup(setup_id, db)
