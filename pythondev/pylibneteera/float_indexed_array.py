# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Type
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

from pylibneteera.math_utils import normal_round

HZ = float  # type alias to clarify that we are working in units of Hertz


class FloatIndexedArray(np.ndarray):
    """ Vectors indexed by floats, in appropriate units """
    __slots__ = ['gap']

    def __new__(cls, input_array: [Type[np.ndarray], list], gap: float) -> 'FloatIndexedArray':
        obj = np.asarray(input_array).view(cls)
        obj.gap = gap
        return obj

    def __reduce__(self):
        array_reduced = super(FloatIndexedArray, self).__reduce__()
        ret = array_reduced[:-1] + (array_reduced[-1] + (self.gap,),)
        return ret

    def __setstate__(self, state):
        try:
            super(FloatIndexedArray, self).__setstate__(state[:-1])
            self.gap = state[-1]
        except TypeError:   # backward compatibility with .npy files that were saved before 19/01/22 change
            super(FloatIndexedArray, self).__setstate__(state)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.gap = getattr(obj, 'gap', None)

    def __getitem__(self, key) -> Union[float, 'FloatIndexedArray']:
        """Special get_item, if passed with an int, acts regular
        >>self[1.5] return the closest value of the array to 1.5 Hz (90 bpm)
        >>self[::n] returns and decimated array with the new correct gap"""
        if isinstance(key, float):
            try:
                return super().__getitem__(int(normal_round(key / self.gap)))
            except IndexError:
                return 0.
        if isinstance(key, slice):
            if key.step:
                out = super().__getitem__(key)
                out.gap *= key.step
                return out
        return super().__getitem__(key)

    def index(self, val: float) -> int:
        """ Return array index corresponding to float value

        :param float val: value to check
        :return: array index of `val`
        :rtype: int
        """
        return int(normal_round(val / self.gap))

    def is_zero(self) -> bool:
        """ Check whether all entries are zero

        :return: `True` if all entries are zero, `False` otherwise
        :rtype: bool
        """
        return np.array([self == 0]).all()


class FrequencyArray(FloatIndexedArray):
    """ Vectors indexed by floats, in Hertz"""
    def zero_out_of_band(self, band):
        """ zero out state and measurement to save computation time, 0^x = 0"""
        freq_array = deepcopy(self)
        freq_array[:self.index(band[0])] = 0
        freq_array[self.index(band[1]):] = 0
        return freq_array

    def fold(self, band: Tuple[HZ, HZ], weights: dict):
        """ Folds a frequency array onto a smaller band of frequencies, by weights. For instance, if the weights
            are `{1: .5, 2: 1., 3: .2}`, a point `p(f)` at frequency `f` is mapped to
            `.5 * p(f) + p(2 * f) + .2 * p(3 * f)`.


        :param Tuple[float, float] band: a band of frequencies to fold onto
        :param  Union[List[float], Dict[float, float]] weights: weights of multiples of frequencies.
        :return: a frequency array onto `band`, weighted by `weights`.
        :rtype: `FrequencyArray`
        """
        gap = self.gap
        first_index_in_band = self.index(band[0])
        last_index_in_band = self.index(band[1]) + 1
        index_range = np.arange(first_index_in_band, last_index_in_band)
        index_mat = np.stack([index_range * i for i in range(1, len(weights) + 1)]).T
        out = np.dot(self[index_mat], list(weights.values()))
        out /= np.max(out)
        out = np.concatenate((np.zeros(first_index_in_band), out, np.zeros(len(self) - last_index_in_band)))
        return FrequencyArray(out, gap)

    def weighted_mean_freq(self, min_freq=0, max_freq=-1) -> HZ:
        """ Frequency at which `self` attains its maximum

        :return: frequency of maximum of `self` in Hz
        :rtype: float
        """
        min_index = self.index(min_freq)
        max_index = -1 if max_freq == -1 else self.index(max_freq)
        return float(np.average(self.freqs[min_index: max_index], weights=self[min_index: max_index]))

    def max_freq(self) -> HZ:
        """ Frequency at which `self` attains its maximum

        :return: frequency of maximum of `self` in Hz
        :rtype: float
        """
        assert len(self.shape) == 1, 'Array must be 1-dimensional'
        return int(np.argmax(self)) * self.gap

    @property
    def freqs(self) -> np.ndarray:
        """:return vector of frequencies [Hz] that correspond to the array (AKA the x-axis of the spectrum)"""
        return np.arange(len(self)) * self.gap

    def plot(self, label=None, units='bpm', new_fig=True, show=True):
        """ plot a FrequencyArray for debugging and research purposes"""
        units_factor = 60 if units == 'bpm' else 1
        if new_fig:
            plt.figure()
        plt.plot(units_factor * self.freqs, self / max(self), label=label)
        plt.xlabel(f'Frequency [{units}]')
        plt.legend()
        plt.xlim([10, 200])
        plt.grid(True)
        if show:
            plt.show()


class TimeArray(FloatIndexedArray):
    def detrend(self):
        return TimeArray(detrend(self), gap=self.gap)

    def duration(self) -> float:
        """duration of the signal in seconds"""
        return len(self) * self.gap

    def retrieve_second(self, sec: int = 1):
        """ Fetch a number of seconds of data from the array, and None if the array is too short

        :param int sec: number of seconds to retrieve
        :return: `sec` seconds of data, or None if the data does not exist
        :rtype: Optional['TimeArray']
        """
        return self[int(sec * self.fs): int((1 + sec) * self.fs)]

    def retrieve_part_of_second(self, start, length):
        """ Fetch a non-whole seconds
        :param start: start time in seconds
        :param length: length ow the window in seconds
        """
        # 1e-6 is Tosefet Shtut (add epsilon so int will not fall by 1 due to finite computation)
        return self[int(start * self.fs): int((length + start + 1e-6) * self.fs)]

    @property
    def fs(self):
        """ sampling frequency (number of samples in a second)"""
        return int(normal_round(1 / self.gap))

    def plot(self, label=None, offset=True, new_fig=True, show=True):
        """ plot a TimeArray for debugging and research purposes"""
        if new_fig:
            plt.figure()
        plt.plot(np.linspace(0, self.gap * len(self), len(self)), self - offset * np.mean(self), label=label)
        plt.xlabel('Time [sec]')
        if show:
            plt.show()
