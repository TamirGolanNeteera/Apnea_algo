from Tests.Plots.PlotReference import plot_ref
from Tests.Utils.LoadingAPI import *
from Tests.Utils.ResearchUtils import plot

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

import keras

from pylibneteera.ExtractPhaseFromCPXAPI import get_fs


def iq_slice(start_sec, end_sec, fs):
    points = 36e3
    start = int(start_sec * fs)
    end = int(end_sec * fs)
    return slice(start, end, int((end - start) / points))


def event_plot_rectangle(signal, value, window_limits, color):
    event_time_points = signal == value
    diff = event_time_points.astype(int).diff()
    start_points = event_time_points.index[np.where(diff == 1)[0]]
    end_points = event_time_points.index[np.where(diff == -1)[0]]
    is_first_of_this_type = True
    for start, end in zip(start_points, end_points):
        if end >= window_limits[0] and start <= window_limits[-1]:
            plt.axvspan(start, end, alpha=0.3, color=color, label=value if is_first_of_this_type else None)
            is_first_of_this_type = False


def plot_iq_apnea_sleep_window(iq, start_ind, end_ind, apnea_sleep_stages, title):
    y_pred = load('/Neteera/Work/homes/moshe.caspi/projects/apnea/results/all_setups/time_str/pred/y_pred_6284.npy')
    y_true = load('/Neteera/Work/homes/moshe.caspi/projects/apnea/results/all_setups/time_str/pred/y_true_6284.npy')
    df = apnea_sleep_stages
    df['y_pred'] = y_pred.droplevel(level=0)
    df['y_true'] = y_true.droplevel(level=0).fillna(np.nan).astype(int)
    df['i'] = iq.iloc[:: get_fs(iq)].apply(np.real).astype(int)
    my_slice = slice(10000, -1, 1)
    axes = df[['i', 'y_pred', 'y_true']].iloc[my_slice].plot(subplots=True)
    plt.sca(axes[0])
    color_dict = {'Obstructive Apnea': 'red', 'Hypopnea': 'blue', 'Central Apnea': 'green', 'Mixed Apnea': 'yellow'}
    for apnea_type, color in color_dict.items():
        event_plot_rectangle(apnea_sleep_stages.apnea, apnea_type, df.index[[0, -1]], color)
    event_plot_rectangle(apnea_sleep_stages.sleep_stages, 'W', df.index[[0, -1]], 'gray')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('I [counts]')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    db = DB()

    setup = 6284
    start_ind = 0
    end_ind = 36000

    apnea_sleep_stages = load_reference(setup, ['apnea', 'sleep_stages'], db)
    try:
        iq = load_nes(setup, db)['data'].iloc[:, 8]
    except IndexError:
        iq = load_nes(setup, db)['data'].iloc[:, 2]
    event_time_points = apnea_sleep_stages.apnea != 'normal'
    diff = event_time_points.astype(int).diff()
    start_indices = np.where(diff == 1)[0]

    title = f'setup {setup}, IQ , apnea and wake data'
    plot_iq_apnea_sleep_window(iq, start_ind, end_ind, apnea_sleep_stages, title)

    y_pred = load('/Neteera/Work/homes/moshe.caspi/projects/apnea/tensor_board/time_str/pred/y_pred_6284.npy')

    for start_ind in start_indices:
        if start_ind > 0:
            plot_iq_apnea_sleep_window(iq, start_ind - 60, start_ind + 120, apnea_sleep_stages, title)
