import logging

from Tests.Utils.TestsUtils import is_debug_mode
from pylibneteera.ExtractPhaseFromCPXAPI import get_fs

from pylibneteera.float_indexed_array import TimeArray, FrequencyArray

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import traceback
import pandas as pd

from pylibneteera.sp_utils import find_peaks_with_modified_prominence

if is_debug_mode():
    matplotlib.use('tkagg')


def get_variable_name(func_name):
    """ extract the variable that passed to the function that calls this function. This has issues in Windows"""
    trace = traceback.extract_stack()
    for frame in trace:
        if func_name in frame.line:
            line = frame.line
            line = line[:line.rfind(')')]
            return line[line.find('(') + 1:]


def print_var(*args):
    vsms_logger = logging.getLogger('vsms')
    msg = f"{get_variable_name('print_var')} = {args if len(args) > 1 else args[0]}"
    if len(vsms_logger.handlers):
        logging.getLogger('vsms').info(msg)
    else:
        print(msg)


def plot_time_signal(signal, gap=None, label=None, offset=False, new_fig=True, show=True, block=False, axe=None,
                     units='sec', grid=True, start_time=None, fig_name=None):
    """ plot a TimeArray for debugging and research purposes"""

    if label is None:
        label = get_variable_name('plot(')
    figure_name = label if fig_name is None else fig_name
    if new_fig and axe is None:
        plt.figure(figure_name)
    if axe is None:
        axe = plt.gca()
    if gap is None:
        fs = get_fs(signal)
        if fs is None:
            gap = 1
            units = 'samples'
        else:
            gap = 1 / fs
    if offset:
        signal -= offset * np.mean(signal)
    if start_time is None:
        x_axis = np.linspace(0, gap * len(signal), len(signal))
    else:
        x_axis = pd.date_range(start=start_time, periods=len(signal), freq=f'{gap}S')
        units = 'clock'
    axe.set_xlabel(f'Time [{units}]')
    axe.plot(x_axis, signal, label=label)
    axe.legend()
    if np.any(np.imag(signal) != 0):
        imag = np.imag(signal)
        axe.plot(x_axis, imag, label=f'{label} imag')
        plt.figure(figure_name + ', i vs q')
        plt.plot(np.real(signal), imag, label=label)
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.legend()
    if grid:
        plt.grid(True)
    if show:
        plt.show(block=block)


def plot_freq_domain(signal, label=None, units='bpm', new_fig=True, show=True, norm_by_max=False,
                     min_freq_to_search_max=0, plot_peaks=True):
    """ plot a FrequencyArray for debugging and research purposes"""

    units_factor = 60 if units == 'bpm' else 1
    sig = signal
    if norm_by_max:
        sig /= max(sig[sig.index(min_freq_to_search_max):])
    if new_fig:
        plt.figure()
    if plot_peaks:
        df = find_peaks_with_modified_prominence(sig, units_factor)
        for i, row in df.iterrows():
            x = row['peaks']
            y = row['peak_heights']
            plt.scatter(x, y, s=50, c='r')
            plt.plot([x, x], [y, y - row['my_prom']], c='g')

    if label is None:
        label = get_variable_name('plot(')
    plt.plot(units_factor * sig.freqs, sig, label=label)
    plt.xlabel(f'Frequency [{units}]')
    plt.legend()
    plt.xlim([10, 200])
    plt.grid(True)
    if show:
        plt.show()


def plot(signal, **kwargs):
    show = kwargs.get('show')
    if (show is None or show) and is_debug_mode():
        matplotlib.use('tkagg')
    if isinstance(signal, FrequencyArray):
        plot_freq_domain(signal, **kwargs)
    else:
        plot_time_signal(signal, **kwargs)


def replace(d: dict, key, val):
    d[key] = val
    return d
