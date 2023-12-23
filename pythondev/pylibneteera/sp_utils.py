# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
import pandas as pd

from pylibneteera.float_indexed_array import TimeArray, FrequencyArray

from scipy.interpolate import interp1d
from scipy.signal import decimate, find_peaks, firwin
from scipy.signal.windows import get_window
from typing import Tuple
from numba import njit
import numpy as np
import logging


@njit()
def fast_lfilter_stride(taps, signal, num_taps, decimation_factor):
    return [np.dot(taps, signal[i: i + num_taps]) for i in range(0, len(signal) - num_taps, decimation_factor)]


def fast_downsample_fir(signal, from_freq: int, to_freq: int, num_taps: int = -1,
                        cutoff: float = -1, decimate_32_bit=False) -> TimeArray:
    """ Downsample an array by using a FIR anti-aliasing filter

    :param np.ndarray signal: The array to be downsampled
    :param int from_freq: frequency of `signal`
    :param int to_freq: frequency to which `signal` is to be downsampled, mod(from_freq, to_freq) must be 0
    :param int num_taps: number of taps for the FIR filter
    :param float cutoff: cutoff frequency for the anti aliasing filter
    :param bool decimate_32_bit: determines if decimation will use 64 data or 32 bit data
    :return: Downsampled array, output frequency, size ceil((len(signal) - num_taps) / int(from_freq // to_freq)
    """
    if len(signal.shape) == 2:
        signal = signal[:, 0]
    if from_freq <= to_freq:
        logging.getLogger('vsms').warning(
            'Cannot upsample. Input frequency: {}. Output frequency: {}'.format(from_freq, to_freq))
        return TimeArray(signal, 1 / from_freq)
    decimation_factor = int(from_freq // to_freq)
    if np.mod(from_freq, to_freq) != 0:
        logging.getLogger('vsms').warning('Input frequency not divisible by 100 Hz! using interpolate')
        n_secs = int(len(signal) // from_freq)
        from_time = np.linspace(0, n_secs, len(signal))
        resampled_time = np.linspace(0, n_secs, n_secs * to_freq * decimation_factor)
        f = interp1d(from_time, signal)
        signal = f(resampled_time)
    if cutoff == -1 or num_taps == - 1:
        num_taps = 20 * decimation_factor + 1
        cutoff = to_freq / 2
    taps = firwin(num_taps, cutoff, fs=from_freq)

    if decimate_32_bit:
        taps = np.asarray(taps, dtype=np.float32)
        if type(signal[0]) is np.float64:
            signal = np.asarray(signal, dtype=np.float32)
        if type(signal[0]) is np.complex128:
            signal = np.asarray(signal, dtype=np.complex64)

    output = fast_lfilter_stride(np.array(taps, dtype=type(signal[0])), signal, num_taps, decimation_factor)
    return TimeArray(output, gap=1 / to_freq)


def downsample(signal, from_freq: int, to_freq: int) -> (np.ndarray, float):
    """ Downsample an array

    :param np.ndarray signal: The array to be downsampled, cast to `tuple` so it is hashable
    :param int from_freq: frequency of `signal`
    :param int to_freq: frequency to which `signal` is to be downsampled
    :return: Downsampled array
    :raises ValueError if from_freq is less than to_freq or if one is negative
    :rtype np.ndarray, float
    """
    def _mirror_resample(sig):
        sig = mirror_pad(sig)
        (samples, channels) = sig.shape
        two_power = int(np.ceil(np.log2(from_freq // to_freq)))
        n_secs = samples // from_freq
        new_sig = np.empty((int(n_secs * to_freq), channels))
        from_time = np.linspace(0, n_secs, len(sig))
        resampled_time = np.linspace(0, n_secs, n_secs * to_freq * 2 ** two_power)
        for c in range(channels):
            f = interp1d(from_time, sig[:, c])
            resampled_sig = f(resampled_time)
            for _ in range(two_power):
                resampled_sig = decimate(resampled_sig, 2)
            new_sig[:, c] = resampled_sig
        return strip_padding(new_sig)

    if len(signal.shape) == 1:
        signal = np.atleast_2d(signal).T
    if to_freq is None or to_freq < 0 or from_freq == to_freq:
        return signal, to_freq
    if from_freq < to_freq:
        logging.getLogger('vsms').warning(
            'Cannot upsample. Input frequency: {}. Output frequency: {}'.format(from_freq, to_freq))
        return signal, from_freq
    if not np.isrealobj(signal):
        return _mirror_resample(np.real(signal)) + 1j * _mirror_resample(np.imag(signal)), to_freq
    return _mirror_resample(signal), to_freq


def resample_sig(data, from_freq, to_freq):
    if from_freq >= to_freq:
        sig = downsample(data, from_freq, to_freq)[0][:, 0]
    else:
        from_time = np.linspace(0, len(data) / from_freq, len(data))
        f = interp1d(from_time, data)
        upsampled_time = np.linspace(0, len(data) / from_freq, int(len(data) * to_freq / from_freq))
        sig = f(upsampled_time)
    return sig


def mirror_pad(array: np.ndarray) -> np.ndarray:
    """ Zero-pad a 1d array by half its length

    :param np.ndarray array: array to be padded
    :return: padded array
    :rtype: np.ndarray
    """
    l_half = array[:int(np.ceil(len(array) / 2))][::-1]
    r_half = array[len(array) // 2:][::-1]
    return np.vstack([l_half, array, r_half])


def normalize(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """ Normalize data by subtracting its mean and dividing by its std

    :param np.ndarray a: array to be normalized
    :param int axis: axis over which to normalize
    :return: a minus its mean divided by its std. If std is zero, divide by 1, and if mean is undefined return a
    :rtype: np.ndarray
    """
    if not list(a):
        return a
    arr_std = np.atleast_1d(np.std(a, axis, dtype=np.float64))
    arr_std[arr_std == 0] = 1  # to divide by 1 for a vector with std zero
    return np.squeeze((a - np.mean(a, axis=axis)) / np.expand_dims(arr_std, axis))


def spectrogram(signal: TimeArray, nfft: int, norm_by_max=False) -> FrequencyArray:
    """returns the freq domain of a signal"""
    window = get_window('hanning', len(signal))
    if nfft is None:
        nfft = len(signal)
    ft = np.fft.rfft(signal * window, n=nfft, axis=0)
    ft = np.abs(ft)[: nfft // 2]
    if norm_by_max:
        ft /= np.max(ft)
    return FrequencyArray(ft, gap=1 / (2 * len(ft) * signal.gap))


def strip_padding(array: np.ndarray) -> np.ndarray:
    """ Cut out the middle half of an array

    :param np.ndarray array: array to be cut
    :return: middle half of array
    :rtype: np.ndarray
    """
    tail = len(array) % 4
    if tail:
        array = array[int(np.floor(tail / 2)): -int(np.ceil(tail / 2))]     # make array divisible by 4
    return array[len(array) // 4:3 * len(array) // 4]


def peak_prominence_in_spectrogram(power: np.ndarray) -> Tuple[float, int]:
    """ Find the best peak's prominence and index.
    :param power: The power spectrogram.

    """
    index_peaks, properties = find_peaks(power, prominence=0.0001, height=0.)
    if len(index_peaks) == 0:
        return 0., 0
    index_max_peak = np.argmax(properties["peak_heights"])
    return properties["prominences"][index_max_peak], index_peaks[index_max_peak]


def max_phase_derivative(phase: TimeArray, to_freq: float) -> float:
    """:return maximum phase of the decimated signal"""
    decimation_factor = int(phase.fs / to_freq)
    # TimeArray.get_item() sets phase_decimated.fs = phase.fs / decimation_factor
    phase_decimated = phase[::decimation_factor]
    return np.max(np.abs(np.diff(phase_decimated))) * phase_decimated.fs    # transfer to units of rad/sec


def zero_crossing(x: iter) -> int:
    """number of times the signal crosses the zero"""
    return ((x[:-1] * x[1:]) < 0).sum()


def find_peaks_with_modified_prominence(sig, units_factor, prominence=0.01, height=0.1, **kwargs):
    peaks, properties = find_peaks(sig, height=height, prominence=prominence, **kwargs)
    properties['peaks_index'] = peaks
    properties['peaks'] = peaks * units_factor * sig.gap
    properties['left_bases_height'] = sig[properties['left_bases']]
    properties['right_bases_height'] = sig[properties['right_bases']]
    df = pd.DataFrame(properties)
    df['my_prom'] = df['peak_heights'] - df[['left_bases_height', 'right_bases_height']].max(axis=1)
    return df