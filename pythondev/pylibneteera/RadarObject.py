import numpy as np

from Configurations import *


class Radar:
    """ A class for radar data.

        The radar class is a convenient container to keep together the measurements,
        derived signals and sampling frequency.

        Do to memory limitations, the data will be truncated (from the front) to a maximum number of seconds.
        Currently, this is configured in Configurations.py and equal to 120 seconds.

    """

    def __init__(self, data: dict, fs: float, truncate: bool = True, truncate_type='hr'):
        """ Construct a radar object.

        Args:
            :param data: The i and q (in-phase and quadrature) measurements in a dictionary.
            :type data: dict
            :param fs: The sampling frequency.
            :type fs: float
            :param truncate: determines if the data will be truncated from the end
            :type truncate: bool
            :param truncate_type: determines if the data will be truncated with hr or hri configuration
            :type truncate: str
        """
        self.data = data  # dictionary with data (i, q, etc)
        self.fs = fs  # sampling frequency

        assert 'i' in self.data.keys(), "No in-phase measurements present in data."
        assert 'q' in self.data.keys(), "No quadrature measurements present in data."
        assert len(self.data['i']) == len(self.data['q']), "No consistent length of measurements."

        # Skip the first `skip_from_start' seconds of the setup.
        skip_from_start = spot_config['setup']['starting_from']  # seconds
        nstart = int(fs * skip_from_start)  # samples

        assert len(self.data['i']) > nstart, "The number of seconds to skip is larger than the setup length."

        self.data['i'] = self.data['i'][nstart:]
        self.data['q'] = self.data['q'][nstart:]

        # Due to memory and time limitations, truncate from the front the data to the maximum number of seconds.
        if truncate:
            if truncate_type == 'hr':
                max_setup_length = spot_config['setup']['maximum_window_from_end']  # seconds
            elif truncate_type == 'bbi':
                max_setup_length = spot_config['setup']['maximum_window_from_end_bbi']  # seconds
            else:
                raise ValueError('Invalid truncate type')
            n_limit = int(fs * max_setup_length)  # samples
            self.data['i'] = self.data['i'][-n_limit:]
            self.data['q'] = self.data['q'][-n_limit:]

    def __dict__(self):
        return {'i': self.data['i'], 'q': self.data['q'], 'fs': self.fs}

    def duration(self) -> int:
        """ Calculate the total duration of the setup, rounded down to the closest second.

        :return: Duration of the setup in seconds.
        :rtype: float
        """
        return int(np.floor(len(self.data['i']) / self.fs))

    def timevec(self):
        return np.arange(self.data['i'].shape[0]) / self.fs

    def demodulate(self):
        """ Demodulate the signal from two measurements in every sample to one.

        The radar produces 'i' and 'q' measurements. To estimate heart rate or respiration,
        this needs to be converted to a one-dimensional signal. Typically the phase is computed,
        but we implement three distinct ways to process the measurements:
        - complex i and q processing;
        - calculating the phase with a static offset;
        - using principal component analysis.
        """
        demodulate_complex_iq(self)  # i + 1im * q
        demodulate_static_offset(self)  # phase with fixed offset
        demodulate_linear_offline(self)  # principal component analysis
        return


# === Demodulation functions ===
def demodulate_static_offset(r: Radar):
    """ Returns the phase the signal.

    It is assumed that the offset is static (which is not always the case).
    Modifies the radar object by storing the returned signal in the r.data dictionary.

    Args:
        :param r: Radar object with measurements in r.data
        :type r: :obj:`Radar`

    Returns:
        :return: the demodulated signal
        :rtype: np.ndarray
    """
    i = r.data['i']
    q = r.data['q']
    new_i = []
    new_q = []
    center_i = (np.max(i) + np.min(i)) / 2
    center_q = (np.max(q) + np.min(q)) / 2

    for kk, _ in enumerate(i):
        new_i.append(i[kk] - center_i)
        new_q.append(q[kk] - center_q)

    phase = np.unwrap(np.arctan2(new_q, new_i))
    r.data['static_offset'] = phase
    return phase


def demodulate_complex_iq(r: Radar):
    """ Returns the signal i + 1j * q, where j = sqrt(-1).

    Modifies the radar object by storing the returned signal in the r.data dictionary.

    Args:
        :param r: Radar object with measurements in r.data
        :type r: :obj:`Radar`

    Returns:
        :return: the demodulated signal
        :rtype: np.ndarray
    """
    x = r.data['i'] + 1j * r.data['q']
    r.data['complex_iq'] = x
    return x


def demodulate_linear_offline(r: Radar):
    """ Returns the first singular vector of the matrix [i q], where i and q are the radar measurements.

    Modifies the radar object by storing the returned signal in the r.data dictionary.

    Args:
        :param r: Radar object with measurements in r.data
        :type r: :obj:`Radar`

    Returns:
        :return: the demodulated signal
        :rtype: np.ndarray
    """
    meas = np.vstack((r.data['i'], r.data['q'])).T
    x = np.linalg.svd(meas, full_matrices=False)
    u = x[0][:, 0]  # first singular vector
    if x[1][0] == 0.:  # The signal is zero everywhere. Return a zero vector.
        y = 0 * u
    else:
        y = u
    r.data['linear_offline'] = y
    return y


def available_demodulation_methods():
    d = {'static_offset': demodulate_static_offset,
         'complex_iq': demodulate_complex_iq,
         'linear_offline': demodulate_linear_offline
         }
    return {k: d[k] for k in spot_config['algo_run_time_tweaks']['demodulation_methods']}


def heart_rate_bandpass_frequencies():
    # Default: [[10., 20.], [10., 30.], [20., 40.], [15., 25.]]
    return spot_config['algo_run_time_tweaks']['high_frequency_bands']  # Hz
