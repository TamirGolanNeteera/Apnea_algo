# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential

from Tests.Plots.PlotRawDataRadarCPX import *

import os
import argparse

from Tests.Utils.LoadingAPI import load_reference


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=True)
    parser.add_argument('-window_length', metavar='TimeWindow', type=int, required=False,
                        help='time window in minutes to display plots')
    plots = ['displacement', 'reference', 'apnea', 'sleep_stages']
    parser.add_argument('-plots', metavar='plots', nargs='+', type=str, help='Plots to create', required=False,
                        default=['displacement', 'apnea'], choices=plots)

    return parser.parse_args()


def plot_displacement_and_rs(timee, displacment, label, idx, fs, start_time=None, end_time=None):

    org_gt_apnea = load_reference(idx, 'apnea', db)
    if db.setup_ts(idx, Sensor.natus)['start'] > db.setup_ts(idx, Sensor.nes)['start']:
        dif = db.setup_ts(idx, Sensor.natus)['start'] - db.setup_ts(idx, Sensor.nes)['start']
        dif = dif.seconds + 12  # 12 seconds difference from the radar or sync
        time_apnea = np.arange(dif * ref_fs, len(org_gt_apnea) + (dif * ref_fs)) / ref_fs
        if start_time is not None:
            if start_time/fs - dif > 0:
                org_gt_apnea = org_gt_apnea[int(start_time/fs - dif) * ref_fs:int(end_time/fs - dif) * ref_fs]
                time_apnea = np.arange(int(start_time/fs) * ref_fs, int(end_time/fs) * ref_fs) / ref_fs
            elif end_time/fs - dif > 0:
                org_gt_apnea = org_gt_apnea[:int(end_time / fs - dif) * ref_fs]
                time_apnea = np.arange(dif * ref_fs, int(end_time/fs) * ref_fs) / ref_fs
            else:
                org_gt_apnea = []
                time_apnea = []
    else:
        dif = db.setup_ts(idx, Sensor.nes)['start'] - db.setup_ts(idx, Sensor.natus)['start']
        dif = dif.seconds - 12
        org_gt_apnea = org_gt_apnea[dif * ref_fs:]
        time_apnea = np.arange(len(org_gt_apnea)) / ref_fs
        if start_time is not None:
            if end_time / fs > time_apnea[-1]:
                org_gt_apnea = org_gt_apnea[int(start_time / fs) * ref_fs:]
                time_apnea = np.arange(int(start_time / fs) * ref_fs, time_apnea[-1] * ref_fs) / ref_fs
            else:
                org_gt_apnea = org_gt_apnea[int(start_time / fs) * ref_fs:int(end_time / fs) * ref_fs]
                time_apnea = np.arange(int(start_time / fs) * ref_fs, int(end_time / fs) * ref_fs) / ref_fs

    fig, ax = plt.subplots(nrows=1)
    if 'displacement' in args.plots:
        ax.plot(timee, displacment, linewidth=0.7, label='NES Displacement')
        ax.set_ylabel(r'$Displacement\ [\mu m]$')
    ax.set_xlabel(r'$Time\ [seconds]$')
    ax.grid(True)
    plt.suptitle(label)
    if 'reference' in args.plots:
        chest_rs = load_reference(idx, 'chest', db)
        nasal_rs = load_reference(idx, 'nasalpress', db)
        if start_time is not None:
            if db.setup_ts(idx, Sensor.natus)['start'] > db.setup_ts(idx, Sensor.nes)['start']:
                if start_time / fs - dif > 0:
                    chest_rs = chest_rs[int(start_time / fs - dif) * ref_fs:int(end_time / fs - dif) * ref_fs]
                    nasal_rs = nasal_rs[int(start_time / fs - dif) * ref_fs:int(end_time / fs - dif) * ref_fs]
                elif end_time / fs - dif > 0:
                    chest_rs = chest_rs[:int(end_time / fs - dif) * ref_fs]
                    nasal_rs = nasal_rs[:int(end_time / fs - dif) * ref_fs]
                else:
                    chest_rs = []
                    nasal_rs = []
            else:
                if end_time / fs > time_apnea[-1]:
                    chest_rs = chest_rs[int(start_time / fs) * ref_fs:]
                    nasal_rs = nasal_rs[int(start_time / fs) * ref_fs:]
                else:
                    chest_rs = chest_rs[int(start_time / fs) * ref_fs:int(end_time / fs) * ref_fs]
                    nasal_rs = nasal_rs[int(start_time / fs) * ref_fs:int(end_time / fs) * ref_fs]
        if len(chest_rs):
            time_apnea = time_apnea[:min(len(time_apnea), len(chest_rs))]
            chest_rs = chest_rs[:min(len(time_apnea), len(chest_rs))]
            nasal_rs = nasal_rs[:min(len(time_apnea), len(chest_rs))]
            ax.plot(time_apnea, 1e8*chest_rs, linewidth=0.7, label='RS Chest')
            ax.plot(time_apnea, 1e11*nasal_rs, linewidth=0.7, label='RS Nasal')

    if 'apnea' in args.plots:
        label_dict = {1: 'hypopnea', 2: 'central apnea', 3: 'obstructive apnea', 4: 'mixed apnea'}
        col = ['blue', 'green', 'red', 'yellow']
        tags = np.unique(org_gt_apnea)
        for n in tags[1:]:
            apn = np.zeros(org_gt_apnea.shape)
            apn[org_gt_apnea == n] = 1
            ver_start = np.where(np.diff(apn) == 1)[0]
            ver_end = np.where(np.diff(apn) == -1)[0]
            if ver_end[0] < ver_start[0]:
                ver_start = np.append(0, ver_start)
            if ver_end[-1] < ver_start[-1]:
                ver_end = np.append(ver_end, len(apn) - 1)
            for v in np.arange(len(ver_start)):
                plt.axvspan(time_apnea[ver_start[v]], time_apnea[ver_end[v]], alpha=0.3, color=col[int(n) - 1],
                            label=label_dict[int(n)] if v == 0 else "")
    if 'sleep_stages' in args.plots:
        sleep_stages = load_reference(idx, 'sleep_stages', db)
        sleep_label_dict = {1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        col = ['blue', 'green', 'red', 'yellow']
        if start_time is not None:
            if db.setup_ts(idx, Sensor.natus)['start'] > db.setup_ts(idx, Sensor.nes)['start']:
                if start_time / fs - dif > 0:
                    sleep_stages = sleep_stages[int(start_time / fs - dif) * ref_fs:int(end_time / fs - dif) * ref_fs]
                elif end_time / fs - dif > 0:
                    sleep_stages = sleep_stages[:int(end_time / fs - dif) * ref_fs]
                else:
                    sleep_stages = []
            else:
                if end_time / fs > time_apnea[-1]:
                    sleep_stages = sleep_stages[int(start_time / fs) * ref_fs:]
                else:
                    sleep_stages = sleep_stages[int(start_time / fs) * ref_fs:int(end_time / fs) * ref_fs]
        tags = np.unique(sleep_stages)
        for n in tags[1:]:
            sleep = np.zeros(sleep_stages.shape)
            sleep[sleep_stages == n] = 1
            ver_start = np.where(np.diff(sleep) == 1)[0]
            ver_end = np.where(np.diff(sleep) == -1)[0]
            if ver_end[0] < ver_start[0]:
                ver_start = np.append(0, ver_start)
            if ver_end[-1] < ver_start[-1]:
                ver_end = np.append(ver_end, len(sleep) - 1)
            for v in np.arange(len(ver_start)):
                plt.axvspan(time_apnea[ver_start[v]], time_apnea[ver_end[v]], alpha=0.3, color=col[int(n) - 1],
                            label=sleep_label_dict[int(n)] if v == 0 else "")
    plt.show(block=False)
    plt.legend(bbox_to_anchor=(1, 1), fontsize='xx-small')
    plt.show()


def process_single_session(winsize, session_id):
    readings, channel_select_params, gap = load_data_from_db(session_id, db)  # load data (setup id from db)
    tot_duration = len(readings) * gap
    if tot_duration <= winsize:
        if session_id is not None:
            print(f'cannot process session {session_id}, total duration {tot_duration} smaller than sliding window'
                  f'{winsize}..')
        raise IndexError
    data = roll_data(readings, channel_select_params, gap, winsize, dont_track=False)
    fs = int(1 / gap)
    phases = np.unwrap(np.angle(data), axis=0)
    time, displacement = gen_disp_time(phases, gap)
    return fs, time, displacement


def plot_single_session(timeee, displacementt, session_id, fs, start_t=None,
                        end_t=None, part=None):
    label = gen_label(False, session_id, '/', db)
    if start_t is not None:
        label += ', part: ' + str(part) + ', start: ' + str(start_t) + ', end: ' + str(end_t)
    plot_displacement_and_rs(timeee, displacementt, label, session_id, fs, start_t, end_t)


if __name__ == '__main__':
    """Displaying tool to show sleep apneas in NWH sessions
    enable showing displacement, references, gt apneas and sleep stages
    enable splitting the sessions display to wanted time periods
    """

    ref_fs = 512
    args = get_args()
    if os.getcwd().endswith('Tests') or os.getcwd().endswith('Tests/'):
        os.chdir('../..')
        print(f'working dir changed to {os.getcwd()}')
    win_size = 10  # HR
    sessions = args.session_ids
    db = DB()
    for sess in sessions:
        try:
            fs, time, displacement = process_single_session(winsize=win_size, session_id=sess)
            if args.window_length:
                start_points = np.arange(0, len(time), step=(args.window_length * 60 * fs))
                for st_pt in range(len(start_points)):
                    if st_pt < len(start_points) - 1:
                        endd = start_points[st_pt + 1] + 300 * fs
                    else:
                        endd = len(time)
                    p_time = time[start_points[st_pt]:endd]
                    p_displacement = displacement[start_points[st_pt]:endd]
                    plot_single_session(p_time, p_displacement, sess, fs,
                                        start_points[st_pt], start_points[st_pt] + len(p_time), st_pt)
            else:
                plot_single_session(time, displacement, sess, fs)

        except (IndexError, UnboundLocalError, TypeError):
            print('Could not process setup {}, skipping...'.format(sess))
        plt.show(block=True)
