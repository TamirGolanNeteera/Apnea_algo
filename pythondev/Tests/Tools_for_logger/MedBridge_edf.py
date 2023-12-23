# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import argparse
import numpy as np
import mne
import os
import pandas as pd
import pytz
import datetime

from mne.io import read_raw_edf

from Tests.vsms_db_api import *

ecg_la = 'ECG-LA'
ecg_ra = 'ECG-RA'
ecg_ll = 'ECG-LL'
nasalPress = 'NasalPress'
chest = 'Chest'
rr = 'RR'
pleth = 'Pleth'
spo2 = 'SpO2'
hr = 'PR'

CITY = "America/New_York"


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-data_location', metavar='Location', type=str, help='location of data')
    parser.add_argument('-benchmark', '-benchamrk', '-benchmarks', '-benchamrks', type=str,
                        choices=[bench.value for bench in Benchmark] + [project.value for project in Project],
                        help='benchmark on which to run version testing, see vsms_db_api.py for more information')
    return parser.parse_args()


def get_start_time(meas_date):
    try:
        start_time = meas_date[0]
    except TypeError:
        start_time = meas_date

    dt_naive = start_time.replace(tzinfo=None)

    tz_info = pytz.timezone(CITY).localize(dt_naive).tzinfo
    return meas_date.replace(tzinfo=tz_info)


def find_gaps(packets):
    diffs = np.diff(packets)
    gap_locations = np.where(diffs != 6)[0]
    return {x: int(diffs[x] - 6 + 0.5) for x in gap_locations}


def edf_to_npy(data_location, just_date=False, save=True):
    folder_path, base_name = os.path.split(data_location)
    base_name = base_name.replace('.edf', '')
    data = read_raw_edf(data_location)
    info = data.info
    fs = int(info['sfreq'])
    start_time = get_start_time(info['meas_date'])

    if just_date:
        return start_time
    names = [x.replace(' ', '_') for x in data.ch_names]
    df = pd.DataFrame({name: data[i][0][0] for i, name in enumerate(names)}, columns=names)
    df.rename(columns=lambda x: x.replace('PulseRate', 'HR'), inplace=True)
    df.set_index(pd.date_range(start_time, periods=len(df), freq=datetime.timedelta(seconds=1/fs)), inplace=True)

    duration = int(len(df)/fs)

    end_time = start_time + datetime.timedelta(seconds=duration)

    epoch_time = start_time.strftime('%s')
    file_names = {col: os.path.join(folder_path, f'{col}_{base_name}_{epoch_time}.npy') for col in df.columns}

    if save:
        for col, filename in file_names.items():
            if col in ['HR', 'SpO2']:
                df.loc[::fs, col].to_pickle(filename)
            else:
                df.loc[:, col].to_pickle(filename)

    return {'start_time': start_time,
            'end_time': end_time,
            'sampling_frequency': fs,
            'duration': duration,
            'filenames': file_names}


if __name__ == '__main__':
    db = DB('neteera_cloud_mirror')
    args = get_args()
    assert args.benchmark or args.data_location
    if args.benchmark:
        for setup in db.benchmark_setups(args.benchmark):
            edf_to_npy(db.setup_ref_path(setup, Sensor.respironics_alice6, search='edf')[0])
    else:
        info_dict = edf_to_npy(args.data_location)
        print(info_dict)





