# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from typing import Callable, Dict, Union, List
import warnings

from biosppy.signals import ecg
import numpy as np
import peakutils
from scipy import interpolate
from scipy.signal import find_peaks
import pandas as pd

from Tests.ReferenceUtils.IEtool import get_ie_rate
from pylibneteera.sp_utils import normalize
from Tests.ReferenceUtils.ecg_peaks import hamilton_segmenter

_EPSILON = np.finfo(np.float16).eps


def biopac_column_formatter(file_path: str) -> Dict[str, int]:
    """ Parse the column number of each indicator, the sampling frequency, and the line from which the data starts
        from a biopac file.

    :param str file_path: path to a biopac file
    :return: the column number of each indicator, the sampling frequency, and the line from which the data starts
    :rtype: Dict[str, int]
    """
    return_dict = {}
    with open(file_path) as f:
        f.readline()
        return_dict['fs'] = int(1000 / int(f.readline().split(' ')[0]))
        channels = int(f.readline().split(' ')[0])
        for i in range(channels):
            field_name = f.readline()
            return_dict[field_name[:-1]] = i + 1
            f.readline()
        return_dict['skiplines'] = 2 * channels + 5
    return return_dict


def neteera_ir_sensor_formatter(channels: int = 16) -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to neteera sensor output

    :param int channels: number of data channels
    :return: callback to return bcg data, reshaped as (number of samples, number of channels)
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def formatter(m: np.ndarray) -> np.ndarray:
        """ BCG data reshaped as (number of samples, number of channels)

          :param np.ndarray m: raw BCG data
          :return: BCG data reshaped as (number of samples, number of channels). Data is truncated if its length is
                   not divisible by channels.
          :rtype: np.ndarray
          """
        try:
            return m.reshape(-1, channels)
        except ValueError:
            print('{} is not divisible by {}. Truncating {} samples.\r\n'.format(len(m), channels, len(m) % channels))
        except AttributeError:
            if m is None:
                raise ValueError('data file is missing')
            raise ValueError('Attempted to load a radar session with an interferometer configuration\r\n')
        return m[:len(m) - len(m) % channels].reshape(-1, channels)

    return formatter


def neteera_sr_sensor_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to SR sensor output

    :return: callback to return SR sensor data, reshaped as (number of samples, number of targets, 3).
            The last dimension represents range,amplitude and phase.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def formatter(m: np.ndarray) -> np.ndarray:
        """ SR sensor data reshaped as (number of samples, targets*3 (range,amplitude,phase))

          :param np.ndarray m: raw SR sensor data
          :return: SR sensor data reshaped as (number of samples, number of targets, 3). The last dimension represents
                    range,amplitude and phase
          :rtype: np.ndarray
          """
        return m

    return formatter


def neteera_sr_cw_sensor_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to SR_CW sensor output

    :return: callback to return SR_CW sensor data, reshaped as (number of samples, number of targets, 3).
            The last dimension represents range,amplitude and phase.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def formatter(m: np.ndarray) -> np.ndarray:
        """ SR sensor data reshaped as (number of samples, targets*3 (range,amplitude,phase))

          :param np.ndarray m: raw SR sensor data
          :return: SR sensor data reshaped as (number of samples, number of targets, 3). The last dimension represents
                    range,amplitude and phase
          :rtype: np.ndarray
          """
        return m

    return formatter


def hexoskin_hr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to hr

    :return: callback to hexoskin heart rate data
    :rtype: Callable Dict[str, List[Union[float, List[float]]]]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Heart rate array

          :param Dict[str, List[Union[float, List[float]]]] data: hexoskin data
          :return: heart rate data each second
          :rtype: np.ndarray
          """
        return np.asarray([60 * data['HR'][i][1] if data['HR'][i][1] != 0 else 0 for i in
                           np.arange(0, len(data['HR']), 2)])
    return formatter


def hexoskin_rr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to rr

    :return: callback to hexoskin respiration rate data
    :rtype: Callable Dict[str, List[Union[float, List[float]]]]
    """

    def resampling(y_vector: List[float], x_vector: List[float], xnew: List[float]):
        """Resample the data to new x axis

        :param List[float] y_vector: y axis values
        :param List[float] x_vector: x axis vector
        :param List[float] xnew:new x axis vector
        :return: resampled y values
        :rtype: List[float]
        """
        x = np.array(x_vector[:len(y_vector)])
        y = np.array(y_vector)
        f = interpolate.interp1d(x, y, fill_value="extrapolate")
        return f(xnew)  # use interpolation function returned by `interp1d`

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Respiration rate array

          :param Dict[str, List[Union[float, List[float]]]] data: data
          :return: respiration rate data each second
          :rtype: List[float]
          """
        time_vector = [i[0] for i in data['RR']]
        new_time_vector = np.arange(0, len(data['RR']))
        rr_old = [i[1] for i in data['RR']]
        rr_vals = resampling(rr_old, time_vector, new_time_vector)

        return np.asarray([i * 60 if i != 0 else 0 for i in rr_vals])

    return formatter


def biopac_hr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to hr

    :return: callback to biopac heart rate data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Heart rate array

          :param np.ndarray data: multicolumn biopac data
          :return: heart rate data each second
          :rtype: np.ndarray
          """
        fs = data['fs']
        try:
            heart_row = data['HR']
        except KeyError:
            heart_row = data['Heart Rate']
        return data['data'][:, heart_row][::fs]

    return formatter


def biopac_rr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to rr

    :return: callback to biopac respiration rate data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Respiration rate array

          :param np.ndarray data: multicolumn biopac data
          :return: respiration rate data each second
          :rtype: np.ndarray
          """
        fs = data['fs']
        respiration_row = data['Respiration Rate']
        return data['data'][:, respiration_row][::fs]

    return formatter


def biopac_inhale_exhale_formatter(win_sec: int) -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to inhale exhale data

    :param int win_sec: seconds per window
    :return: callback to compute inhale exhale, outputted as a complex number of inhale abd exhale times
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Compute inhale exhale

         :param np.ndarray data: multicolumn biopac data
         :return: inhale exhale, outputted as a complex number of inhale abd exhale times
         :rtype: np.ndarray
         """

        def window_inhale_exhale(signal_window, fs) -> float:
            inds = peakutils.indexes(signal_window, thres=0)
            if len(inds) == 0:
                return -1
            relevant_sig = signal_window[inds[0]:inds[-1]]
            diff_of_sig = np.diff(relevant_sig)
            total_inhale_time_locs = np.where(diff_of_sig > 0)[0]
            total_exhale_time_locs = np.where(diff_of_sig < 0)[0]
            total_inhale_time = len(total_inhale_time_locs) / fs    # x_pre.gap
            total_exhale_time = len(total_exhale_time_locs) / fs    # * x_pre.gap
            return total_inhale_time + 1j * total_exhale_time

        fs = data['fs']
        breathing_row = data['Breathing Data']
        signal = data['data'][:, breathing_row]
        inhale_exhale = [window_inhale_exhale(signal[i * fs: (i + win_sec) * fs], fs)
                         for i in range(len(signal) // fs)]
        return np.array(inhale_exhale)

    return formatter


def biopac_ecg_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to ecg waveform

    :return: callback to biopac ecg data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ ECG array

          :param np.ndarray data: multicolumn biopac data
          :return: ECG data each sample
          :rtype: np.ndarray
          """
        ecg_row = data['ECG Raw']
        return data['data'][:, ecg_row]

    return formatter


def biopac_breathing_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to breathing waveform

    :return: callback to biopac breathing data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: multicolumn biopac data
          :return: breathing data each sample
          :rtype: np.ndarray
          """
        breathing_row = data['Breathing Data']
        return data['data'][:, breathing_row]

    return formatter


def biopac_rra_formatter(win_sec: int) -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to respiration amplitude

    :param int win_sec: seconds per window
    :return: callback to compute respiration amplitude, computed as the std of the breathing column
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Compute respiration amplitude array

         :param np.ndarray data: multicolumn biopac data
         :return: respiration amplitude each second, computed as the std of the breathing column
         :rtype: np.ndarray
         """

        def reject_outliers(dataa, m: int = 2):
            dataa = dataa.astype(np.float64)
            return dataa[abs(dataa - np.mean(dataa)) <= m * np.std(dataa)]

        def window_respiration_amplitude(signal_window) -> float:
            top_inds = peakutils.indexes(signal_window, min_dist=int(fs * 1.5))
            bottom_inds = peakutils.indexes(-1 * signal_window, min_dist=int(fs * 1.5))
            if len(top_inds) and len(bottom_inds):
                top_peak_vec = signal_window[top_inds]
                bottom_peak_vec = signal_window[bottom_inds]
            else:
                return 0
            if np.mean(reject_outliers(top_peak_vec)) - np.mean(reject_outliers(bottom_peak_vec)) < 0:
                warnings.warn("RA is negative")
                return -1
            return np.mean(reject_outliers(top_peak_vec)) - np.mean(reject_outliers(bottom_peak_vec))

        fs = data['fs']
        breathing_row = data['Breathing Data']
        signal = data['data'][:, breathing_row]
        respiration_amplitude = [window_respiration_amplitude(signal[i * fs: (i + win_sec) * fs])
                                 for i in range(len(signal) // fs - win_sec)]
        return np.array(respiration_amplitude)

    return formatter


def ecg_formatter(fs: int, win_sec: int) -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :param int fs: sampling frequency of signal
    :param int win_sec: seconds per window
    :return: callback to compute hr, computed as average interval between peaks in signal
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def compute_rate(signal: np.ndarray) -> np.ndarray:
        """ Compute hr array

        :param np.ndarray signal: ECG signal for which hr is computed
        :raises ZeroDivisionError when Signal is constant or there is only one peak
        :return: hr each second, computed as average interval between peaks in signal
        :rtype: np.ndarray
        """
        signal = normalize(np.squeeze(signal))
        win_hri = []
        for i in range(signal.shape[0] // fs):
            rpeaks = np.array(ecg.hamilton_segmenter(signal[i * fs:(i + win_sec) * fs], fs))
            if abs(np.mean(np.ediff1d(rpeaks[0]))) < _EPSILON:
                raise ZeroDivisionError('Only one peak in all of ecg')
            win_hri.append(60. / (np.mean(np.ediff1d(rpeaks[0])) / fs))
        return np.array(win_hri, dtype=np.float16)

    assert fs > 0
    return compute_rate


def imotion_biopac_hr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :return: callback to compute hr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: imotion_biopac samples
          :return: hr of imotion_biopac samples
          :rtype: np.ndarray
          """
        return np.array(data['ECG HR'])

    return formatter


def imotion_biopac_rr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to rr computation

    :return: callback to compute rr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: imotion_biopac samples
          :return: rr of imotion_biopac samples
          :rtype: np.ndarray
          """
        return np.array(data['RSP'])

    return formatter


def capnograph_pc900b_hr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :return: callback to compute hr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: capnograph_pc900b samples
          :return: rr of capnograph_pc900b samples
          :rtype: np.ndarray
          """
        return np.repeat(data['Pulse(bmp)'].to_list(), 4)  # sample every 4 sec

    return formatter


def capnograph_pc900b_rr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to rr computation

    :return: callback to compute rr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: capnograph_pc900b samples
          :return: rr of capnograph_pc900b samples
          :rtype: np.ndarray
          """
        return np.repeat(data['Respiratory Rate(bmp)'].to_list(), 4)

    return formatter


def epm_10m_hr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :return: callback to compute hr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: epm_10m samples
          :return: hr of epm_10m samples
          :rtype: np.ndarray
          """
        try:
            return np.array([float(d) if d.isdigit() else -1.0 for d in data.hrbpm])
        except AttributeError:
            return np.array([float(d) if d else -1.0 for d in data.hrbpm])

    return formatter


def epm_10m_rr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to rr computation

    :return: callback to compute rr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: epm_10m samples
          :return: rr of epm_10m samples
          :rtype: np.ndarray
          """
        try:
            return np.array([float(d) if d.isdigit() else -1.0 for d in data.rrrpm])
        except AttributeError:
            return np.array([float(d) if d else -1.0 for d in data.rrrpm])

    return formatter


def epm_10m_bp_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to bp computation

    :return: callback to compute bp
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: epm_10m samples
          :return: bp of epm_10m samples
          :rtype: np.ndarray
          """
        try:
            bp = []
            for i in range(len(data.nibpdmmhg)):
                try:
                    bp.append(int(data.nibpsmmhg[i]) + 1j * int(data.nibpdmmhg[i]))
                except ValueError:
                    pass
            if len(bp) == 0:
                return np.array(0) + 1j * np.array(0)
            else:
                return np.array(bp)
        except ValueError:
            return np.array(0) + 1j * np.array(0)
    return formatter


def epm_10m_peaks_formatter():
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        signal = []
        for row in data:
            signal += list(row)[1:]
        fs = len(data[0]) - 1    # output sampling frequency of Mindray - epm_10m
        return np.array(hamilton_segmenter(np.array(signal), fs)[0],
                        dtype=np.int64) * 1000 // fs  # convert from sec to millisec
    return formatter


def epm_10m_ie_formatter(data):
    co2 = pd.DataFrame(data).to_numpy()
    co2 = co2[:, 1:].flatten()
    return get_ie_rate(co2)


def elbit_ecg_sample_hr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :return: callback to compute hr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: elbit ecg sample
          :return: hr of elbit ecg sample
          :rtype: np.ndarray
          """
        return data.hr

    return formatter


def elbit_ecg_sample_rr_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to rr computation

    :return: callback to compute rr
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: elbit ecg sample
          :return: rr of elbit ecg sample
          :rtype: np.ndarray
          """
        return data.rr

    return formatter


def elbit_ecg_sample_bp_formatter() -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to bp computation

    :return: callback to compute bp
    :rtype: Callable[[np.ndarray], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: elbit ecg sample
          :return: bp of elbit ecg sample
          :rtype: np.ndarray
          """
        try:
            try:
                bp = data.bp[0].split('/')
            except AttributeError:
                return np.array(0) + 1j * np.array(0)
        except IndexError:
            bp = data.bp.tolist().split('/')
        return np.array(int(bp[0])) + 1j * np.array(int(bp[1]))

    return formatter


def bitalino_formatter(win_sec: int) -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hr computation

    :param int win_sec: seconds per window
    :return: callback to compute hr, computed as average interval between peaks in signal
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def compute_rate(signal: np.ndarray) -> np.ndarray:
        """ Compute hr array

        :param np.ndarray signal: ECG signal for which hr is computed
        :raises ZeroDivisionError when Signal is constant or there is only one peak
        :return: hr each second, computed as average interval between peaks in signal
        :rtype: np.ndarray
        """
        fs = signal['fs']
        signal = normalize(np.squeeze(signal['data']))
        win_hri = []
        for i in range(signal.shape[0] // fs):
            rpeaks = np.array(ecg.hamilton_segmenter(signal[i * fs:(i + win_sec) * fs], fs))
            if abs(np.mean(np.ediff1d(rpeaks[0]))) < _EPSILON:
                raise ZeroDivisionError('Only one peak in all of ecg')
            win_hri.append(60. / (np.mean(np.ediff1d(rpeaks[0])) / fs))
        return np.array(win_hri, dtype=np.float16)

    return compute_rate


def ecg_variance_formatter(fs: int, win_sec: int) -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hrv computation

    :param int fs: sampling frequency of signal
    :param int win_sec: seconds per window
    :return: callback to compute hrv, computed as std of intervals between peaks in signal
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def compute_std(signal: np.ndarray) -> np.ndarray:
        """ Compute hrv array

        :param np.ndarray signal: ECG signal for which hr is computed
        :raises ZeroDivisionError when Signal is constant or there is only one peak
        :return: hrv, computed as stds of intervals between peaks in signal
        :rtype: np.ndarray
        """
        signal = normalize(np.squeeze(signal))
        win_hri = []
        for i in range(signal.shape[0] // fs):
            rpeaks = np.array(ecg.hamilton_segmenter(signal[i * fs:(i + win_sec) * fs], fs))
            if abs(np.mean(np.ediff1d(rpeaks[0]))) < _EPSILON:
                raise ZeroDivisionError('Only one peak in all of ecg')
            if len(rpeaks[0]) > 2:
                hri = np.std(np.ediff1d(rpeaks[0]) / fs) * 1000  # convert from sec to millisec
                if not np.isnan(hri):
                    win_hri.append(hri)
            else:
                win_hri.append(-1)
        return np.array(win_hri, dtype=np.float16)

    assert fs > 0
    return compute_std


def dn_ecg_interval_formatter(fs: int) -> Callable[[np.ndarray], np.ndarray]:
    """ Return callback to hri computation

    :param int fs: sampling frequency of signal
    :return: callback to compute hri, computed as intervals between peaks of the entire signal
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    def compute_interval(signal: np.ndarray) -> np.ndarray:
        """ Compute hri array

        :param np.ndarray signal: ECG signal for which hri is computed
        :raises ZeroDivisionError when Signal is constant or there is only one peak
        :return: hri, computed as intervals between peaks of the entire signal
        :rtype: np.ndarray
        """
        signal = normalize(np.squeeze(signal))
        peaks_ecg, properties_ecg = find_peaks(signal, prominence=None, width=None, distance=round(fs / 2))
        hri_ecg = (np.ediff1d(peaks_ecg) / fs) * 1000  # convert to msec
        if abs(np.mean(hri_ecg)) < _EPSILON:
            raise ZeroDivisionError('Only one peak in all of ecg')
        return hri_ecg
    return compute_interval


def biopac_peaks_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to hr peaks computation

    :return: callback to compute hr peaks, computed as the tempotal location of the peaks in signal
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Compute hr peaks array

          :param np.ndarray data: multicolumn biopac data
          :raises ZeroDivisionError when Signal is constant or there is only one peak
          :return: hr peaks, computed as the tempotal location of the peaks in signal
          :rtype: np.ndarray
          """
        ecg_row = data['ECG Raw']
        signal = data['data'][:, ecg_row]
        fs = 250
        return np.array(ecg.hamilton_segmenter(signal, fs)[0],
                        dtype=np.float16) * 1000 // fs  # convert from sec to millisec
    return formatter


def zephyr_hr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to hr

    :return: callback to biopac heart rate data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def smooth(x, window_len=10, window='flat'):
        """
        :param x: input signal
        :param window_len: window size
        :param window: smoothing algorithm
        :return:  smoothed input signal
        """
        # Code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def formatter(data:  Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Heart rate array

          :param np.ndarray data: multicolumn biopac data
          :return: heart rate data each second
          :rtype: np.ndarray
          """

        return smooth(np.array(data['HR'], dtype=np.float16))

    return formatter


def zephyr_rr_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to rr

    :return: callback to biopac respiration rate data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Respiration rate array

          :param np.ndarray data: multicolumn biopac data
          :return: respiration rate data each second
          :rtype: np.ndarray
          """

        return np.array(data['BR'], dtype=np.float16)

    return formatter


def zephyr_ecg_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to ecg waveform

    :return: callback to biopac ecg data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ ECG array

          :param np.ndarray data: multicolumn biopac data
          :return: ECG data each sample
          :rtype: np.ndarray
          """
        return np.array(data[' ECG Data'], dtype=np.float16)

    return formatter


def zephyr_breathing_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to breathing waveform

    :return: callback to biopac breathing data
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Breathing array

          :param np.ndarray data: multicolumn biopac data
          :return: breathing data each sample
          :rtype: np.ndarray
          """

        return np.array(data[' Breathing Waveform'], dtype=np.float16)

    return formatter


def zephyr_peaks_formatter() -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to hrv computation

    :return: callback to compute hrv, computed as std of intervals between peaks in signal
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """
    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Compute hrv array

          :param np.ndarray data: multicolumn biopac data
          :raises ZeroDivisionError when Signal is constant or there is only one peak
          :return: hrv, computed as stds of intervals between peaks in signal
          :rtype: np.ndarray
          """
        signal = zephyr_ecg_formatter()(data)
        fs = 252
        return np.array(ecg.hamilton_segmenter(signal, fs)[0],
                        dtype=np.float16) * 1000 // fs  # convert from sec to millisec
    return formatter


def zephyr_rra_formatter(win_sec: int) -> Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]:
    """ Return callback to respiration amplitude

    :param int win_sec: seconds per window
    :return: callback to compute respiration amplitude, computed as the std of the breathing column
    :rtype: Callable[[Dict[str, Union[int, np.ndarray]]], np.ndarray]
    """

    def formatter(data: Dict[str, Union[int, np.ndarray]]) -> np.ndarray:
        """ Compute respiration amplitude array

         :param np.ndarray data: multicolumn biopac data
         :return: respiration amplitude each second, computed as the std of the breathing column
         :rtype: np.ndarray
         """

        def reject_outliers(dataa, m: int = 2):
            dataa = dataa.astype(np.float64)
            return dataa[abs(dataa - np.mean(dataa)) <= m * np.std(dataa)]

        def window_respiration_amplitude(signal_window) -> float:
            top_inds = peakutils.indexes(signal_window, min_dist=int(fs * 1.5))
            bottom_inds = peakutils.indexes(-1 * signal_window, min_dist=int(fs * 1.5))
            if len(top_inds) and len(bottom_inds):
                top_peak_vec = signal_window[top_inds]
                bottom_peak_vec = signal_window[bottom_inds]
            else:
                return 0
            if np.mean(reject_outliers(top_peak_vec)) - np.mean(reject_outliers(bottom_peak_vec)) < 0:
                warnings.warn("RA is negative")
                return -1
            return np.mean(reject_outliers(top_peak_vec)) - np.mean(reject_outliers(bottom_peak_vec))

        fs = 18
        signal = zephyr_breathing_formatter()(data)
        respiration_amplitude = [window_respiration_amplitude(signal[i * fs: (i + win_sec) * fs])
                                 for i in range(len(signal) // fs - win_sec)]
        return np.array(respiration_amplitude)

    return formatter


def spirometer_ie_formatter(data):
    pass


def natus_posture_formatter(df):
    output = np.empty(int(df.onset.iloc[-1]), dtype='<U20')
    df = df[df.description.apply(lambda x: 'Body Position: ' in x)]
    if len(df) == 0:
        return output
    onset = df['onset']
    df['end'] = list(onset.iloc[1:]) + [len(output)]
    df.loc[df.index[0], 'onset'] = 0
    for _, row in df.iterrows():
        state = row['description'].replace('Body Position: ', '')
        for index in range(int(row['onset']), int(row['end'])):
            output[index] = state
    return output


def natus_apnea_formatter(df):
    output = ['normal'] * int(df.onset.iloc[-1])
    df = df[df.description.apply(lambda x: 'pnea' in x)]
    if len(df) == 0:
        return output
    df['end'] = df['onset'] + df['duration']
    df.loc[df.index[-1], 'end'] = min(df['end'].iloc[-1], len(output))
    for _, row in df.iterrows():
        start = int(row['onset'])
        for index in range(start, int(row['end'])):
            output[index] = row['description']
    return output


def natus_sleep_stages_formatter(df):
    output = np.empty(int(df.onset.iloc[-1]), dtype='<U20')
    df = df[df.description.apply(lambda x: 'Sleep stage ' in x)]
    if len(df) == 0:
        return output
    onset = df['onset']
    df['end'] = list(onset.iloc[1:]) + [len(output)]
    df.loc[df.index[0], 'onset'] = 0
    for _, row in df.iterrows():
        state = row['description'].replace('Sleep stage ', '')
        for index in range(int(row['onset']), int(row['end'])):
            output[index] = state
    return output
