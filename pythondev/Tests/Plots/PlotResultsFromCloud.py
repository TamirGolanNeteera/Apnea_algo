# script that takes a folder of results and raw data and plots

import os
import sys

from Tests.Plots.PlotResults import Y_LIM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from Tests.Utils.PandasUtils import pd_str_plot
from Tests.Utils.LoadingAPI import load, load_reference
from Tests.Utils.TestsUtils import *

import argparse
import matplotlib.pyplot as plt
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

VS_SUPPORTED = ['hr', 'rr', 'bbi', 'stat', 'ie']

Y_MIN = {'hr': 39, 'rr': 1, 'stat': -1, 'bbi': 1, 'ie': -1, 'spo2': 80}
Y_MAX = {'hr': 160, 'rr': 50, 'stat': 4, 'bbi': 2000, 'ie': 4, 'spo2': 101}


def get_args() -> argparse.Namespace:
    """ Parse arguments"""
    parser = argparse.ArgumentParser(description='Plot algorithm output versus ground truth')
    parser.add_argument('-setups', '-session_ids', '-setup_ids', '-setup', nargs='+', type=int, help='setup list')
    parser.add_argument('-load_path', '-result_dir', '-folder_list', metavar='load_path', type=str,
                        help='Path to load files from')
    parser.add_argument('-save_fig_path', metavar='Location', type=str, required=False,
                        help='location of output figures')
    return parser.parse_args()


def find_endswith(list_strings, ending):
    for x in list_strings:
        if x.endswith(ending):
            return x
    print('Warning: no string ends with ending:', ending)


def plot_visualization(device_sn):
    plt.ylim(Y_LIM['hr'])
    plt.xlabel('Time [Jerusalem]')
    plt.grid()
    plt.ylabel('HR [bpm]')
    plt.title(f"Subject-007.1-NC-F3-R3-001 SN {device_sn}")


def plot_single_file(load_path, save_fig_path, ref):
    list_dir = os.listdir(load_path)

    pred_csv_path = os.path.join(load_path, find_endswith(list_dir, 'VS.csv'))
    pred_df = pd.read_csv(pred_csv_path)

    raw_data_path = os.path.join(load_path, find_endswith(list_dir, 'cloud_raw_data.json'))
    raw_dict = load(raw_data_path)

    # pred_df['bin'] = [x['bins']for x in raw_dict['seconds'][1:]]

    df = pred_df[['hr', 'rr', 'stat']]
    axes = pd_str_plot(df[df.stat.notna()])
    axes[0].set_ylim([Y_MIN['hr'], Y_MAX['hr']])
    axes[1].set_ylim([Y_MIN['rr'], Y_MAX['rr']])

    if save_fig_path is None:
        plt.show()
    else:
        plt.savefig(save_fig_path)

    ref_hr = ref.astype(int)
    ref_hr[ref_hr < 1] = np.nan
    ref_hr.index = (ref_hr.index + pd.to_timedelta(7, unit='H')).time
    ref_hr.name = 'ref'

    pred_start = datetime.datetime.fromtimestamp(pred_df.ts[0]).replace(tzinfo=pytz.timezone('Asia/Jerusalem'))
    pred_hr = pred_df.hr.astype(int)
    pred_hr.index = pd.date_range(start=ref.index[0].astimezone(pytz.timezone('Asia/Jerusalem')), freq='S',
                                  periods=len(pred_df)).time
    pred_hr[pred_hr < 1] = np.nan
    pred_hr.name = 'pred'

    df = pd.concat([pred_hr, ref_hr], axis=1).rename(columns={'hr': 'pred', 'HR': 'ref'})
    df.plot()
    plot_visualization(raw_dict['session_metadata']['device_sn'])


def main_plot_results(arguments):
    db = DB('neteera_cloud_mirror')
    if arguments.setups:
        for setup in arguments.setups:
            ref = load_reference(setup, 'hr', db)
            path = os.path.dirname(db.setup_ref_path(setup, Sensor.nes)[0])
            plot_single_file(path, arguments.save_fig_path, ref)


if __name__ == "__main__":
    main_plot_results(get_args())
