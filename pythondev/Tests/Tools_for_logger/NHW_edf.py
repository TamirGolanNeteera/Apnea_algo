# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import argparse
import numpy as np
import mne
import os
import pandas as pd
import pytz
import datetime

from Tests.Tools_for_logger.edf_neteeera import read_raw_edf

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
    parser.add_argument('-data_location', metavar='Location', type=str, required=True, help='location of data')
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
    data = read_raw_edf(data_location, preload=True)
    info = data.info
    fs = int(info['sfreq'])
    start_time = get_start_time(info['meas_date'])

    if just_date:
        return start_time

    df = pd.DataFrame({name: data[i][0][0] for i, name in enumerate(data.ch_names)}, columns=data.ch_names)
    df.drop(['RR'], axis=1, inplace=True)
    df.rename(columns=lambda x: x.replace('PR', 'HR'), inplace=True)

    for gap_loc, gap_len in find_gaps(data.packets).items():
        df = pd.concat([df.iloc[:gap_loc * fs],
                        pd.DataFrame(index=range(gap_len * fs), columns=df.columns),
                        df.iloc[gap_loc * fs:]])
    df.set_index(pd.date_range(start_time, periods=len(df), freq=pd.Timedelta(nanoseconds=1e9/fs)), inplace=True)

    duration = int(len(df)/512)

    end_time = start_time + datetime.timedelta(seconds=duration)

    annotation_df = pd.DataFrame(data.annotations.__dict__)
    annotation_filename = os.path.join(folder_path, f'annotations_{base_name}.csv')

    epoch_time = start_time.strftime('%s')
    file_names = {col: os.path.join(folder_path, f'{col}_{base_name}_{epoch_time}.npy') for col in df.columns}

    if save:
        for col, filename in file_names.items():
            if col in ['HR', 'SpO2']:
                df.loc[::int(info['sfreq']), col].to_pickle(filename)
            else:
                df.loc[:, col].to_pickle(filename)
        annotation_df.to_csv(annotation_filename, index=False)

    file_names['annotation'] = annotation_filename

    return {'start_time': start_time,
            'end_time': end_time,
            'sampling_frequency': info['sfreq'],
            'duration': duration,
            'filenames': file_names}


if __name__ == '__main__':
    args = get_args()
    info_dict = edf_to_npy(args.data_location)
    print(info_dict)





