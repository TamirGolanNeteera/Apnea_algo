from Tests.vsms_db_api import *

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import datetime
from os import listdir
from os.path import isfile, join
import argparse
import Configurations
from pylibneteera.math_utils import np_normal_round

REF_FS = 40  # The reference FS


def parse_args():
    """
    parsing arguments from user
    """
    parser = argparse.ArgumentParser(description='Get arguments to plot')
    parser.add_argument('-folder_path', metavar='folder_path', type=str,
                        help='The location of the file on the server', required=False)
    return parser.parse_args()


def get_ie_ratio_spirometer_batch(volume, fs):
    n_sample = len(volume)
    lp_filter = sps.butter(Configurations.front_chair_config['ie']['lowpass_filter_degree'],
                           [2 / 60, Configurations.front_chair_config['ie']['lowpass_filter_cutoff']],
                           btype='bandpass', fs=fs)
    smoothed_phase = sps.filtfilt(lp_filter[0], lp_filter[1], volume)

    # derivative and threshold
    phase_derivative = np.hstack((200 * np.diff(smoothed_phase), 1))
    threshold = Configurations.front_chair_config['ie']['threshold_factor'] * np.asarray(
        [np.std(phase_derivative[i:i + int(fs)]) for i in range(n_sample - int(fs))])

    # inhale / exhale / nohale sections
    exhale = np.where(phase_derivative[:len(threshold)] <= -threshold, smoothed_phase[:len(threshold)], np.nan)
    inhale = np.where(phase_derivative[:len(threshold)] >= threshold, smoothed_phase[:len(threshold)], np.nan)
    nohale = np.where(np.logical_and(np.isnan(inhale), np.isnan(exhale)), smoothed_phase[:len(threshold)], np.nan)

    state = np.where(np.isnan(inhale), 0, 0) + np.where(np.isnan(exhale), 0, 1) + np.where(np.isnan(nohale), 0, 2)

    # run length encoding: [2 2 2 3 3 1 1 1 1 1] -> [2 3 1], [3 2 5]
    starts = np.r_[0, np.where(~np.isclose(state[1:], state[:-1], equal_nan=True))[0] + 1]
    points = np.diff(np.r_[starts, len(state)])
    breathing_state = state[starts]

    # find valid breathing patterns
    pattern = [2, 0, 2, 1, 2]
    valid_patterns = [np.all(pattern == x) for x in
                      [breathing_state[i:i + len(pattern)] for i in range(len(breathing_state) - len(pattern))]]

    # find starting indices of patterns / inhales / exhales
    indx = np.nonzero(valid_patterns)
    valid_inhale = points[indx[0] + 1] / fs
    valid_exhale = points[indx[0] + 3] / fs

    # compute moving averages, ratio, and rounded i.e.-ratios
    moving_average_inhale = np.convolve(np.ones(4) / 4, valid_inhale, mode='same')
    moving_average_exhale = np.convolve(np.ones(4) / 4, valid_exhale, mode='same')

    ratios = np_normal_round((moving_average_inhale / moving_average_exhale), 1)
    result = np.median(ratios)
    return result


def get_ie_ratio_spirometer(vol, fs):
    n = Configurations.front_chair_config['ie']['win_sec']
    num_sec = int(np.round(len(vol) / fs) - 1 - n)
    ratios = [get_ie_ratio_spirometer_batch(vol[(i*fs):((i+n)*fs)], fs) for i in range(num_sec)]
    return np.hstack([-np.ones(n-1), np.asarray(ratios)])


def get_ie_rate(data: np.ndarray, window_size: int = 20, gap_size: int = 1, fs=REF_FS) -> np.ndarray:
    """
    Inhale Exhale rate calculation
    @param data: input array of filtered displacement / CO2
    @param window_size: size of window we do calculations on
    @param gap_size: the gap between beggining of windows
    @param fs: frames per second
    @return: ndarray of the IE rate per window
    """
    ie_rate = []
    for i in range(0, len(data), gap_size * fs):
        window = data[i: i + window_size * fs]
        peaks, _ = find_peaks(window, distance=150)
        if len(peaks) > 1:
            start, end = peaks[0], peaks[-1]
            cut_window = window[start:end + 1]
            derivative = np.diff(cut_window)
            curr_rate = np.sum(derivative < 0) / np.sum(derivative > 0)
            ie_rate.append(curr_rate)
        else:
            ie_rate.append(-1)
    return np.array(ie_rate)


def delay_by_ts(pred_ts, gt_ts):
    """
    Gets the delay
    @param pred_ts: timestamp array of prediction data
    @param gt_ts: timestamp array of ground truth data
    @return: the delay in seconds between prediction to ground truth
    """
    first_element = datetime.datetime.strptime(gt_ts[0], "%Y-%m-%d %H:%M:%S")
    first_unix_gt = int(datetime.datetime.timestamp(first_element))
    first_pred_ts = int(np.round(pred_ts[0] / 1000))
    return int(first_pred_ts - first_unix_gt)


def get_pred_gt_ie(setup_num, folder_path, is_csv=False):
    """
    Gets the prediction and ground truth Inhale/Exhale ratios from data
    @param setup_num:
    @param folder_path:
    @param is_csv:
    @return:
    """
    if is_csv:
        pred = pd.read_csv(os.path.join(folder_path, str(setup_num) + '.csv'))
        pred_ie = pred['ie'].to_numpy()
    else:
        pred_path = db.setup_ref_path(setup=setup_num, sensor=Sensor.nes_res)[0]
        pred = pd.read_csv(pred_path)
        pred_ie = np.load(join(folder_path, str(setup_num) + '_ie.npy'), allow_pickle=True)
    gt_path = db.setup_ref_path(setup_num, sensor=Sensor.epm_10m, search='CO2')[0]
    gt = pd.read_csv(gt_path, header=None).to_numpy()
    delta = delay_by_ts(pred['ts'], gt.T[0])
    gt = gt[:, 1:].flatten()[delta * REF_FS:]
    gt_ie = get_ie_rate(gt)
    gt_ie = gt_ie[~np.isnan(gt_ie)]
    pred_ie = pred_ie[:len(gt_ie)]
    gt_ie, pred_ie = gt_ie[pred_ie > -1], pred_ie[pred_ie > -1]
    return pred_ie, gt_ie


def plot_single_setup(pred_ie, gt_ie, folder_path, setup_num):
    """
    Saves plot for single setup I/E ratio
    @param pred_ie:
    @param gt_ie:
    @param folder_path:
    @param setup_num:
    @return:
    """
    plt.plot(np.arange(len(pred_ie)), pred_ie, label='Neteera I/E', color='tab:blue')
    plt.plot(np.arange(len(gt_ie)), gt_ie, label='Reference I/E on CO2', color='tab:orange')
    plt.legend()
    plt.xlabel("Second")
    plt.ylabel("I/E Ratio")
    plt.title("Inhale / Exhale ratio for setup " + str(setup_num))  # todo: get setup num somehow
    plt.grid()
    plt.ylim([0, 4])
    plt.xlim([0, len(gt_ie) - 1])
    plt.fill_between(range(len(gt_ie)), gt_ie * 0.9,
                     gt_ie * 1.1, alpha=0.75, label='REF margin', color='wheat')
    plt.savefig(join(folder_path, str(setup_num) + '_ie_plot.png'))
    plt.close()


def generate_ie_plots(folder_path):
    ie_files = [file for file in listdir(folder_path) if isfile(join(folder_path, file))
                and 'ie' in file and file.endswith('npy')]
    cpp_files = [file for file in listdir(folder_path) if isfile(join(folder_path, file)) and file.endswith('.csv')]
    for fname in ie_files:
        setup_num = int(fname.split('_')[0])
        try:
            pred_ie, gt_ie = get_pred_gt_ie(setup_num, folder_path)
            plot_single_setup(pred_ie, gt_ie, folder_path, setup_num)
        except IndexError:
            print('setup ' + str(setup_num) + ' not found, skipping...')
            continue
    for fname in cpp_files:
        setup_num = int(fname.split('.')[0])
        try:
            pred_ie, gt_ie = get_pred_gt_ie(setup_num, folder_path, True)
            plot_single_setup(pred_ie, gt_ie, folder_path, setup_num)
        except IndexError:
            print(f'setup {setup_num} not found, skipping...')
            continue


if __name__ == '__main__':
    db = DB()
    args = parse_args()
    if args.folder_path is None:
        exit('Usage error. Please enter folder path as argument like this:\n'
             '-folder_path *your/path/to/ie_files.npy*')
    generate_ie_plots(args.folder_path)
