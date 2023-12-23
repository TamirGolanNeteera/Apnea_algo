import sys
import os

from Tests.Utils.PathUtils import path_dirname_multiple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # noqa

from Configurations import back_chair_config

from pylibneteera.filters import Filter
from pylibneteera.float_indexed_array import TimeArray
from pylibneteera.math_utils import normal_round

import ctypes
from numpy import ctypeslib as ctl
import numpy as np
import pandas as pd
import platform
from scipy import optimize


def get_calc_R_func(x, y):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    return calc_R


def get_f_2_func(calc_R):
    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    return f_2


def _set_circle_fit():
    if platform.system() == 'Linux':  # parallel linux - qsub runs
        lib_name = 'CircleFit.so'
    else:
        if '64bit' in platform.architecture():
            lib_name = 'CircleFit_64.dll'
        else:
            lib_name = 'CircleFit_32.dll'
    try:
        lib = ctl.load_library(lib_name, path_dirname_multiple(__file__, 2))
    except OSError:
        return
    return lib.circleFitDll


def get_fs(signal, fs=None):
    if fs is not None:
        return fs
    if isinstance(signal, TimeArray):
        return signal.fs
    try:
        return normal_round(1/signal.index.freq.delta.total_seconds())
    except (TypeError, AttributeError, IndexError):
        return


class PhaseAPI:
    def __init__(self):
        self.circle_fit = _set_circle_fit()
        self.filter_obj = Filter()
        self.config = back_chair_config

    def _find_circle_center_dll(self, x_vec: np.ndarray, y_vec: np.ndarray):
        """ calculates the center with a dll and returns it as [x_center, y_center]"""
        iq_vec = np.stack((x_vec, y_vec), -1)
        self.circle_fit.argtypes = [
            ctl.ndpointer(np.float64, shape=(len(x_vec), 2), flags='ALIGNED, C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int,
            ctl.ndpointer(np.float64, shape=(1, 2), flags='ALIGNED, C_CONTIGUOUS')]
        ret = np.array([[0, 0]], dtype=np.float64)
        self.circle_fit(iq_vec, len(x_vec), 2, ret)
        if np.any(np.isnan(ret)):
            print('circle fit dll returns nan!!!!!!!!!!!')
            return None
        return np.squeeze(ret)

    @staticmethod
    def _find_circle_center_python(x_vec: np.ndarray, y_vec: np.ndarray):
        """ calculates the center without a dll and returns it as [x_center, y_center]"""
        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x_vec-xc)**2 + (y_vec-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = [0, 0]
        center_2, ier = optimize.leastsq(f_2, center_estimate)
        return center_2

    def _find_circle_center(self, x_vec: np.ndarray, y_vec: np.ndarray, use_dll=True):
        """ calculates the center with a dll and returns it as [x_center, y_center]"""
        if use_dll:
            return self._find_circle_center_dll(x_vec, y_vec)
        else:
            return self._find_circle_center_python(x_vec, y_vec)

    def calc_offset(self, complex_signal, fs, to_freq, use_dll):
        downsampled_complex = self.filter_obj.fast_downsample_fir(np.array(complex_signal), fs, to_freq=to_freq)
        if len(downsampled_complex) < 10:
            downsampled_complex = complex_signal
        real = np.real(downsampled_complex)
        imag = np.imag(downsampled_complex)
        return self._find_circle_center(real, imag, use_dll)

    def iq_to_phase(self, complex_signal, input_fs=None, reduce_offset=True, to_freq=10, use_dll=False):
        """ converts iq data to phase data

        :param complex_signal: complex I + jQ iterable
        :param input_fs: input frequency sampling rate, if the data is TimeArray the function will use
        complex_signal.fs. If the data is Time-Series then the fs will be 1/increment
        :param reduce_offset: Whether or not to reduce offset
        :param to_freq: Decimation of the signal before passing through circle fit to reduce run-time
        :param use_dll: The dll is used in python-to-embedded versions for bit exact
        :return: the phase (angle) vector in radians
        """
        fs = get_fs(complex_signal, input_fs)
        off_set = self.calc_offset(complex_signal, fs, to_freq, use_dll) if reduce_offset else [0, 0]
        offset_reduced = np.array(complex_signal) - off_set[0] - 1j * off_set[1]
        angles = pd.Series(np.angle(offset_reduced)).fillna(0)
        output_data = np.unwrap(angles, axis=0)
        if isinstance(complex_signal, pd.Series):
            return pd.Series(output_data, index=complex_signal.index)
        else:
            return TimeArray(output_data, gap=1/fs)


if __name__ == '__main__':
    example_data = np.cos(np.linspace(0, 15, 14999)) + 1j*np.sin(np.linspace(0, 15, 14999)) + 4 - 5j
    print(PhaseAPI().iq_to_phase(example_data, input_fs=500))
