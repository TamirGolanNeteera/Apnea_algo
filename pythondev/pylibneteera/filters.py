# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, lfilter, firwin
from typing import Tuple, Union

from pylibneteera.float_indexed_array import TimeArray
from pylibneteera.sp_utils import fast_lfilter_stride


class Filter:
    """ wrapper class that holds the filter constants to reduce run time"""
    def __init__(self):
        self.filters = {}
        self.taps_fir = {}

    def _butter_bandpass(self, low, high, order, pass_type):
        """ Return the relevant Butterworth filter

        :return: Butterworth filter
        :rtype: Tuple[np.ndarray, Union[np.ndarray, float], np.ndarray]
        """
        if (low, high, order, pass_type) in self.filters:
            return self.filters[(low, high, order, pass_type)]
        if not 0 < high < 1:
            output = butter(order, low, btype='high', output=pass_type)
        elif low <= 0:
            output = butter(order, high, output=pass_type)
        else:
            output = butter(order, [low, high], btype='bandpass', output=pass_type)
        self.filters[(low, high, order, pass_type)] = output
        return output

    def butter_bandpass_filter(self, data: TimeArray, lowcut: float, highcut, order: int = 2,
                               pass_type: str = 'filtfilt', lfilter_reduct_tail=None) -> np.ndarray:
        """ Performs a Butterworth bandpass filter, or a lowpass or a highpass when appropriate."""
        nyq = .5 * data.fs
        low, high = lowcut / nyq, highcut / nyq
        if pass_type == 'lfilter':
            b, a = self._butter_bandpass(low, high, order, 'ba')
            out = lfilter(b, a, data)[int(len(data) * lfilter_reduct_tail):]
        elif pass_type == 'filtfilt':
            sos = self._butter_bandpass(low, high, order, 'sos')
            out = sosfiltfilt(sos, data)
        else:
            raise Exception('invalid filter pass_type')
        return TimeArray(out, 1 / data.fs)

    def fast_downsample_fir(self, signal, from_freq: int, to_freq: int, num_taps: int = -1,
                            cutoff: float = -1) -> TimeArray:
        """ Downsample an array by using a FIR anti-aliasing filter

        :param np.ndarray signal: The array to be downsampled
        :param int from_freq: frequency of `signal`
        :param int to_freq: frequency to which `signal` is to be downsampled, mod(from_freq, to_freq) must be 0
        :param int num_taps: number of taps for the FIR filter
        :param float cutoff: cutoff frequency for the anti aliasing filter
        :return: Downsampled array, output frequency, size ceil((len(signal) - num_taps) / int(from_freq // to_freq)
        """
        if len(signal.shape) == 2:
            signal = signal[:, 0]
        if from_freq <= to_freq:
            logging.getLogger('vsms').warning(
                'Cannot up-sample. Input frequency: {}. Output frequency: {}'.format(from_freq, to_freq))
            return TimeArray(signal, 1 / from_freq)
        decimation_factor = int(from_freq // to_freq)
        if np.mod(from_freq, to_freq) != 0:
            logging.getLogger('vsms').warning(
                f'Input frequency {from_freq} Hz is not divisible by output frequency {to_freq} Hz! using interpolate')
            n_secs = int(len(signal) // from_freq)
            from_time = np.linspace(0, n_secs, len(signal))
            resampled_time = np.linspace(0, n_secs, n_secs * to_freq * decimation_factor)
            f = interp1d(from_time, signal)
            signal = f(resampled_time)
        if cutoff == -1 or num_taps == - 1:
            num_taps = 20 * decimation_factor + 1
            cutoff = to_freq / 2
        if (num_taps, cutoff, from_freq) in self.taps_fir:
            taps = self.taps_fir[(num_taps, cutoff, from_freq)]
        else:
            taps = firwin(num_taps, cutoff, fs=from_freq)

        output = fast_lfilter_stride(np.array(taps, dtype=type(signal[0])), signal, num_taps, decimation_factor)
        return TimeArray(output, gap=1 / to_freq)
