# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from Tests.Constants import UNITS
from Tests.Utils.StringUtils import join
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import *
from Tests.Plots.PlotResults import Y_MIN, Y_MAX
from Tests.Utils.PandasUtils import pd_str_plot, get_gap_from_time_series

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd


def get_args() -> argparse.Namespace:
    """ Parse arguments"""
    parser = argparse.ArgumentParser(description='Plot algorithm output versus ground truth')
    parser.add_argument('-result_dir', '-folder_list', '-save_fig_path', metavar='load_path', type=str, required=False,
                        help='Path to load files from')
    parser.add_argument('-setups', '-session_ids', metavar='ids', nargs='+', type=int)
    vs = [str(x) for x in VS]
    parser.add_argument('-vital_sign', '-compute', nargs='+', type=str, choices=vs, default=vs,
                        help='plot the following vital signs (default is all)\n')
    return parser.parse_args()


def plot_ref(vs, idx, ref, axe=None):
    if axe is None:
        plt.figure(f'reference {vs}  setup {idx} ')
        axe = plt.gca()
    plt.sca(axe)
    axe.set_title(f'reference {vs}  setup {idx} ')
    axe.set_ylabel(f'{vs} [{UNITS[vs]}]' if vs in UNITS else vs)
    axe.grid()
    plt.yticks(rotation=45)
    if isinstance(ref, pd.Series):
        gap = get_gap_from_time_series(ref)
        plt.plot(np.linspace(0, len(ref) * gap, len(ref)), ref.fillna('NA').to_list())
        plt.show(block=False)
        bottom_ax = plt.gca()
        upper_ax = bottom_ax.twiny()  # creates the upper ticks
        x1, x2 = bottom_ax.get_xbound()
        y1, y2 = upper_ax.get_xbound()
        m = (y2 - y1) / (x2 - x1)
        x_tick_loc = [m * (x - x1) + y1 for x in bottom_ax.get_xticks()[1:-1]]
        upper_ax.set_xticks(x_tick_loc)
        start_time = ref.index[0]
        upper_ax.set_xticklabels([(start_time + datetime.timedelta(seconds=x)).strftime('%H:%M') if len(ref) > x >= 0
                                  else ''
                                  for x in bottom_ax.get_xticks()[1:-1]])
        axe.set_xlabel('Time [sec]')
    else:
        axe.plot(list(ref))
        axe.set_xlabel('Time [samples]')
    if vs in Y_MIN:
        plt.ylim([Y_MIN[vs], Y_MAX[vs]])


def load_plot_and_save_ref(idx, vs, folder=None):
    ref = load_reference(idx, vs, db)
    if ref is None:
        return
    plot_ref(vs, idx, ref)
    if folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, f"reference_{vs}_{join(db.setup_multi(idx), '_')}.png"))
        plt.clf()
        plt.close()


if __name__ == "__main__":
    db = DB()
    args = get_args()
    if args.result_dir is not None:
        matplotlib.use('agg')
    for setup in args.setups:
        for vital_sign in args.vital_sign:
            load_plot_and_save_ref(setup, vital_sign, args.result_dir)
