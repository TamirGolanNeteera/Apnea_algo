# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
import os
from functools import wraps
from time import time
from typing import Callable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# from Tests.Plots.PlotAccumulatedFrequencyDomain import create_single_video
from Tests.Utils.PathUtils import create_dir

GLOBAL_IN_MEMORY_LOGGING = {}
GLOBAL_IN_MEMORY_LOGGING_TIMES = {}
SAVE_PATH_KEY = 'save_path'
TIME = {'time': 0}


def set_time(t):
    TIME['time'] = t


def accumulate_return_values(func: Callable) -> Callable:
    """ Accumulate return values of a function in the global GLOBAL_IN_MEMORY_LOGGING dictionary

    :param Callable func: function being wrapped
    :return: The return value for the function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if func in GLOBAL_IN_MEMORY_LOGGING:
            GLOBAL_IN_MEMORY_LOGGING[func][TIME['time']] = res
        else:
            GLOBAL_IN_MEMORY_LOGGING[func] = pd.Series({TIME['time']: res})
        return res
    return wrapper


def accumulate_execution_times(func: Callable) -> Callable:
    """ Accumulate execution times of a function in the global GLOBAL_IN_MEMORY_LOGGING dictionary.
        If running a whole setup, appends the setup identifier to the runtime.

    :param Callable func: function being wrapped
    :return: The return value for the function
    :rtype: Callable
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = time()
        _ = func(*args, **kwargs)
        if SAVE_PATH_KEY in kwargs.keys():
            save_path = kwargs[SAVE_PATH_KEY]
            setup = kwargs['idx']
        else:  # for parallel run where kwargs does not exist
            save_path = args[2]
            setup = args[1]
        create_dir(os.path.join(save_path, 'accumulated'))
        np.save(os.path.join(save_path, 'accumulated', str(setup) + '_times'), np.array([0, time() - s]))
    return wrapper


def dump_accumulated_values(path: str, setup, save_video=False):
    """ Dump values accumulated in GLOBAL_IN_MEMORY_LOGGING to files
    """
    global GLOBAL_IN_MEMORY_LOGGING
    acc_path = os.path.join(path, 'accumulated')
    create_dir(acc_path)
    for k in GLOBAL_IN_MEMORY_LOGGING:
        accumulated_list = GLOBAL_IN_MEMORY_LOGGING[k]
        file_name = os.path.join(path, 'accumulated', f'{setup}{k.__name__}.npy')
        accumulated_list.to_pickle(file_name)
    GLOBAL_IN_MEMORY_LOGGING = {}


def plot_accumulated_values():
    """ Plot values accumulated in GLOBAL_IN_MEMORY_LOGGING """
    for i, k in enumerate(GLOBAL_IN_MEMORY_LOGGING):
        accumulated_list = GLOBAL_IN_MEMORY_LOGGING[k]
        if not isinstance(accumulated_list[0], (int, float, complex)):
            print(f'Cannot plot {k.__name__}, not a number: {accumulated_list[:10]}')
            continue
        plt.figure(i)
        try:
            plt.title(f'{k.__name__} - accumulated values')
        except AttributeError:
            plt.title(f'{k} - accumulated values')
        plt.plot(accumulated_list)
    plt.show()
