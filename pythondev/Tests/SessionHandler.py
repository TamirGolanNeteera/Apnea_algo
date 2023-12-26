# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from Configurations import *

from pylibneteera.datatypes import NeteeraSensorType, SensorType

import logging
import os
import re
from csv import DictReader
from typing import Dict, List, Optional, Union, IO
import mne
import numpy as np
import pandas as pd

from pylibneteera.filters import Filter
from pylibneteera.float_indexed_array import TimeArray

VALID_TARGETS = 16
IR_CHANNELS = 16
HEXOSKIN_KEYS = ['ecg_peaks', 'ecg_raw', 'HR', 'HRV', 'respiration', 'RR']


def retrieve(data_paths: Dict[SensorType, str]) -> Dict[SensorType,
                                                        Union[np.ndarray, Dict[str, np.ndarray]]]:
    """ Retrieve data for sensors at the specified file paths

    :param Dict[SensorType, str]) data_paths: File paths for data files for requested sensors
    :raises FileNotFoundError if a file does not exist
            PermissionError if the file is locked
            ValueError if the file has an invalid extension
    :return: The data for each requested sensor, None if no path is provided
    :rtype: Dict[SensorType, Union[np.ndarray, Dict[str, Union[int, np.ndarray]]]]
    """
    return_dict = {}
    for d in data_paths:
        if data_paths[d] is not None:
            return_dict[d] = _load(data_paths[d].replace('"', ''), d)
    return return_dict


def bcg_filename(metadata: np.record) -> str:
    """ Return filename of the BCG file

    :param np.record metadata: metadata of setup
    :return: filename of BCG file
    :rtype: str
    """
    return os.path.splitext(metadata.base_filename)[0]


def gt_type(metadata: np.record) -> Optional[SensorType]:
    """ The type of ground truth recorded for the file

    :param np.record metadata: setup metadata
    :return: The type of ground truth recorded for the file, or None if there is no ground truth
    :rtype: Optional[SensorType]
    """
    lookup_dict = {SensorType.biopac: metadata.biopac,
                   SensorType.ecg: metadata.ecg,
                   SensorType.gps: metadata.gps,
                   SensorType.spO2: metadata.spo2,
                   SensorType.bitalino: metadata.bitalino,
                   SensorType.natus: metadata.natus}

    gt_type_list = [a for a in lookup_dict if len(lookup_dict[a]) > 0]
    if not gt_type_list:
        return None
    return gt_type_list[0]


def parse_zephyr_file(data_path: str, type: str = "") -> Dict[str, Union[int, np.ndarray]]:
    return_dict = {}
    with open(data_path) as f:
        # read the file as a dictionary for each row ({header : value})
        reader = DictReader(f)
        if type != "":
            for i in range(5):
                reader.fieldnames[i] = type + "_" + str(reader.fieldnames[i])
        for row in reader:
            for header, value in row.items():
                return_dict.setdefault(header, []).append(value)
    return return_dict


def _load(data_path: str, data_type) -> Optional[Union[np.ndarray, Dict[str, Union[int, np.ndarray]]]]:
    """ Load data from a sensor

    :param str data_path: File path for the data
    :param data_type: Type of sensor from which the data is collected
    :raises FileNotFoundError if a file does not exist
            PermissionError if the file is locked
            ValueError if the file has an invalid extension
    :return: array of sensor data or None if the array is not found. Dict of sensor data with metadata for biopac
    :rtype:  Optional[Union[np.ndarray, Dict[str, Union[int, np.ndarray]]]]
    """
    if data_path == '':
        logging.getLogger('vsms').warning('No path provided')
        return None
    try:
        SENSOR_TYPE_LOADERS = {NeteeraSensorType.fmcw: _load_fmcw,
                               SensorType.biopac: _load_biopac,
                               SensorType.zephyr_hr_rr: _load_zephyr_hr_rr,
                               SensorType.zephyr_bw: _load_zephyr_bw,
                               SensorType.zephyr_ecg: _load_zephyr_ecg,
                               SensorType.stat: _load_array,
                               SensorType.ecg: _load_ecg,
                               SensorType.gps: _load_gps,
                               SensorType.spO2: _load_spo2,
                               SensorType.bitalino: _load_bitalino,
                               SensorType.natus: _load_natus,
                               SensorType.dn_ecg: _load_dn_ecg,
                               SensorType.hexoskin: _load_hexoskin,
                               SensorType.elbit_ecg_sample: _load_elbit_ecg_sample,
                               SensorType.imotion_biopac: _load_imotion_biopac,
                               SensorType.epm_10m: _load_epm_10m,
                               SensorType.capnograph_pc900b: _load_capnograph_pc900b}
        return SENSOR_TYPE_LOADERS[data_type](data_path)

    except FileNotFoundError:
        logging.getLogger('vsms').warning(f'The file does not exist: {data_path}')
    except PermissionError:
        logging.getLogger('vsms').warning(f"You don't have the permission to open the file: {data_path}")
    return None


def _load_cw_sensor_data(in_file: IO) -> Dict[str, Union[float, Dict[str, float]]]:
    """ Load SR sensor data

    :param io in_file: handle to data file
    :return: dict of SR sensor data
    :rtype: Dict[str, Union[float, Dict[str, float]]]
    """
    ret_dict = {'data': {'I': [], 'Q': []},
                'agc_fit': 0,
                'agc_sat': 0}
    duration = 0
    framerate = 0
    agc_disabled = False
    rgx_iq = r'^[A-Z][A-Z][A-Z]:'
    rgx_duration = r'^duration\Wms\W=([0-9]*)$'
    rgx_framerate = r'^FW\sframerate='
    rgx_agc_fit = r'AGC_GainFitMonitorDur='
    rgx_agc_disabled = r'AGC=disabled'
    for line in in_file:
        if re.compile(rgx_iq, re.IGNORECASE).search(line) is not None:
            try:
                i, q = [int(a) for a in line.split(':')[-1].split(',')[:-1]]
            except ValueError:
                i, q, dt = [int(a) for a in line.split(':')[-1].split(',')[:-1]]
                try:
                    ret_dict['data']['dt'].append(dt / 10)  # dt = time diff from prev. sample in 1/10 micro sec;
                    # convert to micro sec and collect
                except KeyError:
                    ret_dict['data']['dt'] = [dt / 10]
            ret_dict['data']['I'].append(i)
            ret_dict['data']['Q'].append(q)
        elif re.compile(rgx_duration, re.IGNORECASE).search(line) is not None:
            duration = int(re.compile(r'\d+').findall(line)[0])
        elif re.compile(rgx_framerate, re.IGNORECASE).search(line) is not None:
            framerate = int(re.compile(r'\d+').findall(line)[0])
        elif re.compile(rgx_agc_fit, re.IGNORECASE).search(line) is not None:
            ret_dict['agc_fit'], ret_dict['agc_sat'] = int(re.compile(r'\d+').findall(line)[0]), \
                                                       int(re.compile(r'\d+').findall(line)[1])
        elif re.compile(rgx_agc_disabled, re.IGNORECASE).search(line) is not None:
            agc_disabled = True
        else:
            pass
    if framerate != 0:
        ret_dict['framerate'] = framerate
    elif duration != 0:
        ret_dict['framerate'] = (1000 * len(ret_dict['data']['I'])) / duration  # convert to ms
    if agc_disabled:
        ret_dict['agc_fit'] = 0
        ret_dict['agc_sat'] = 0
    return ret_dict


def process_raw(sig, nfft, high_pass=False):
    filter_object = Filter()
    rng_fft = []
    for frame_sig in sig:
        if high_pass:
            frame_sig = filter_object.butter_bandpass_filter(TimeArray(frame_sig, gap=1e-6), 20e3, 1e100)
        rng_fft.append(np.fft.fft(frame_sig, n=nfft))
    rng_mat = np.vstack(rng_fft)
    return rng_mat


def process_cpx(sig):
    rng_mat = np.vstack(sig)
    return rng_mat


def _load_fmcw_sensor_data(in_file: IO) -> Dict[str, Union[float, str, np.ndarray, Dict[str, Union[str, list, float]]]]:
    """ Load SR sensor data

    :param io in_file: handle to data file
    :return: dict of SR sensor data
    :rtype: Dict[str, Union[float, str, Dict[str, Union[str, list, float]]]
    """
    rgx_sig = r'^[A-Z][A-Z][A-Z]:'
    rgx_framerate = r'^FW\sframerate='
    rgx_bw = r'^Bandwidth*'
    rgx_raw = r'RAW_ADC'
    rgx_bins = r'bin_offset'
    rgx_sample_rate = r'ADC_clkDiv'
    rgx_adc_samplesNum = r'ADC_samplesNum'
    rgx_pre_buf = r'PreBuf'
    rgx_post_buf = r'PostBuf'
    rgx_fft_size = r'FFT_size'

    def read_raw_from_file(data_file):
        fs = 0.
        bw = 0
        is_raw = False
        signals = []
        bins = np.array([])
        sr = 0
        adc_samplesNum = 0
        pre_buf = 0
        post_buf = 0
        nfft = None

        for line in data_file:
            if re.compile(rgx_sig, re.IGNORECASE).search(line) is not None:
                iq_data = line.split(':')[1].strip()[:-1].split(',')
                iq_data = np.array(list(map(int, iq_data)))
                signals.append(np.dot(iq_data.reshape(-1, 2), np.array([1, 1j])))
            elif re.compile(rgx_framerate, re.IGNORECASE).search(line) is not None:
                fs = int(re.compile(r'\d+').findall(line)[0])
            elif re.compile(rgx_bw, re.IGNORECASE).search(line) is not None:
                bw = int(re.compile(r'\d+').findall(line)[0])
            elif re.compile(rgx_raw, re.IGNORECASE).search(line) is not None:
                is_raw = bool(int(line.split('=')[-1]))
            elif re.compile(rgx_sample_rate, re.IGNORECASE).search(line) is not None:
                sr = float(re.compile(r'\d+\.?\d*').findall(line)[0])
            elif re.compile(rgx_adc_samplesNum, re.IGNORECASE).search(line) is not None:
                adc_samplesNum = int(re.compile(r'\d+\.?\d*').findall(line)[0])
            elif re.compile(rgx_pre_buf, re.IGNORECASE).search(line) is not None:
                pre_buf = int(re.compile(r'\d+\.?\d*').findall(line)[0])
            elif re.compile(rgx_post_buf, re.IGNORECASE).search(line) is not None:
                post_buf = int(re.compile(r'\d+\.?\d*').findall(line)[0])
            elif re.compile(rgx_fft_size, re.IGNORECASE).search(line) is not None:
                nfft = int(re.compile(r'\d+\.?\d*').findall(line)[2])
            elif re.compile(rgx_bins, re.IGNORECASE).search(line) is not None:
                first_bin = int(re.compile(r'\d+').findall(line)[1])
                num_bins = int(re.compile(r'\d+').findall(line)[0])
                bins = np.arange(first_bin, first_bin + num_bins)
        return signals, fs, bw, is_raw, bins, sr, adc_samplesNum, pre_buf, post_buf, nfft

    signal, framerate, bandwidth, raw, data_bins, sample_rate, adc_samplesNum, pre_buf, post_buf, nfft = \
        read_raw_from_file(in_file)
    if nfft is None:
        nfft = back_chair_config['fmcw_raw_params']['nfft']
    try:
        processed_signal = process_raw(signal, nfft) if raw else process_cpx(signal)
    except ValueError:
        logging.getLogger('vsms').warning('setup %s NO DATA\r\n', in_file.name)
        raise ValueError

    ret_dict = {'data': processed_signal,
                'framerate': framerate,
                'BW': bandwidth,
                'is_raw': raw,
                'raw': signal if raw else None,
                'bins': np.arange(nfft) if raw else data_bins,
                'sample_rate': sample_rate,
                'PLLConfig_PreBuf': pre_buf,
                'PLLConfig_PostBuf': post_buf,
                'basebandConfig_ADC_samplesNum': adc_samplesNum,
                'basebandConfig_FFT_size': nfft}
    return ret_dict


def _load_fmcw(path: str) -> Dict[str, Union[np.ndarray, float, Dict[str, np.ndarray]]]:
    filename, file_extension = os.path.splitext(path)
    base_path = os.path.basename(path).replace(file_extension, '')
    tlog_files = [os.path.join(os.path.dirname(path), f) for f in
                  os.listdir(os.path.dirname(path)) if (f.endswith(file_extension) and f.find(base_path) >= 0)]
    order = -1 * np.ones(len(tlog_files))
    for i in range(len(tlog_files)):
        try:
            order[i] = int(tlog_files[i].split('.')[-2])
        except ValueError:
            order[i] = 0
    sorted_tlog_files = list(np.asarray(tlog_files)[np.argsort(order)])
    return_dict = {}
    for f_iter, data_path in enumerate(sorted_tlog_files):
        with open(data_path, encoding="Latin-1") as f:
            cur_data = _load_fmcw_sensor_data(in_file=f)
            if f_iter == 0:
                return_dict = cur_data
            else:
                if 'framerate' in cur_data.keys() and cur_data['framerate'] > 0:
                    return_dict['framerate'] = cur_data['framerate']
                if 'sample_rate' in cur_data.keys() and cur_data['sample_rate'] > 0:
                    return_dict['sample_rate'] = cur_data['sample_rate']
                if return_dict['is_raw']:
                    return_dict['data'] = np.append(
                        return_dict['data'], process_raw(cur_data['data'], cur_data['basebandConfig_FFT_size']), axis=0)
                else:
                    return_dict['data'] = np.append(return_dict['data'], process_cpx(cur_data['data']), axis=0)

    return return_dict


def _load_array(data_path: str) -> np.ndarray:
    """ Load results array

    :param str data_path: path to data
    :raises ValueError if file extension is invalid
    :return: array of results
    :rtype: np.ndarray
    """
    extension = str(os.path.splitext(data_path)[1])
    if extension == '.npy':
        return np.load(data_path)
    if str(extension).lower() == '.txt':
        return np.loadtxt(data_path)
    raise ValueError('Invalid file extension')


def _load_elbit_ecg_sample(data_path: str) -> np.ndarray:
    """ Load elbit ecg sample data

    :param str data_path: path to data
    :return: array of elbit ecg sample data
    :rtype: np.ndarray
    """
    return np.recfromcsv(data_path, encoding=None)


def _load_imotion_biopac(data_path: str) -> np.ndarray:
    """ Load imotion_biopac sample data

    :param str data_path: path to data
    :return: array of imotion_biopac sample data
    :rtype: np.ndarray
    """
    return pd.read_excel(data_path, skiprows=1)


def _load_epm_10m(data_path: str) -> np.ndarray:
    """ Load epm_10m sample data

    :param str data_path: path to data
    :return: array of epm_10m sample data
    :rtype: np.ndarray
    """
    try:
        return np.recfromcsv(data_path, encoding=None)
    except ValueError as e:
        print(e)
        print(f'could not load the last row in file {data_path}')
        return np.recfromcsv(data_path, encoding=None, skip_footer=1)


def _load_capnograph_pc900b(data_path: str) -> np.ndarray:
    """ Load capnograph_pc900b sample data

    :param str data_path: path to data
    :return: array of capnograph_pc900b sample data
    :rtype: np.ndarray
    """
    return pd.read_excel(data_path)


def _load_spo2(data_path: str) -> np.ndarray:
    """ Load spO2 data

    :param str data_path: path to data
    :raises ValueError if file extension is invalid
    :return: array of spO2 data
    :rtype: np.ndarray
    """
    # .csv files have keys 'pulse' and 'spo2', and some need "\"" stripped.
    # .txt files from MHE do not
    extension = str(os.path.splitext(data_path)[1])
    if extension == '.csv':
        def double_quote(x):
            return int(x.strip("\""))
        return np.recfromcsv(data_path, encoding=None, converters={"pulse": double_quote,
                                                                   "spo2": double_quote})['pulse']
    if extension == '.txt':
        return np.genfromtxt(data_path, dtype=None, autostrip=True, delimiter="\t", encoding=None)


def _load_bitalino(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load data from biopac

    :param str data_path: path to data
    :return: biopac data
    :rtype: np.ndarray
    """
    return_dict = {'fs': 1000}
    skip_rows = 3
    return_dict['data'] = np.loadtxt(data_path, skiprows=skip_rows, usecols=5, dtype=np.float16)
    return return_dict


def _load_dn_ecg(data_path: str) -> np.ndarray:
    """ Load ECG data

    :param str data_path: path to data
    :raises ValueError if file extension is invalid
    :return: array of ECG data
    :rtype: np.ndarray
    """
    extension = str(os.path.splitext(data_path)[1])
    if extension == '.csv':
        return np.recfromcsv(data_path, encoding='unicode_escape', skip_header=2).ecg
    raise ValueError("Bad file extension")


def _load_ecg(data_path: str) -> np.ndarray:
    """ Load ECG data

    :param str data_path: path to data
    :raises ValueError if file extension is invalid
    :return: array of ECG data
    :rtype: np.ndarray
    """
    extension = str(os.path.splitext(data_path)[1])
    if extension == '.csv':
        return np.loadtxt(data_path)
    if extension == '.dat' or '.txt':
        return np.loadtxt(data_path)[:, 1]
    raise ValueError("Bad file extension")


def _load_biopac(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load data from biopac

    :param str data_path: path to data
    :return: biopac data
    :rtype: np.ndarray
    """
    return_dict = {}
    with open(data_path) as f:
        f.readline()
        return_dict['fs'] = int(1000 / float(f.readline().split(' ')[0]))
        channels = int(f.readline().split(' ')[0])
        for i in range(channels):
            field_name = f.readline()
            return_dict[field_name[:-1]] = i + 1
            f.readline()
        skip_rows = 2 * channels + 5
        return_dict['data'] = np.loadtxt(data_path, skiprows=skip_rows, dtype=np.float16)
    return return_dict


def _load_zephyr_hr_rr(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load hr rr data from zephyr

    :param str data_path: path to data
    :return: zephyr data
    :rtype: np.ndarray
    """
    return parse_zephyr_file(data_path)


def _load_zephyr_bw(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load breathing wave data from zephyr

    :param str data_path: path to data
    :return: zephyr data
    :rtype: np.ndarray
    """
    return parse_zephyr_file(data_path)


def _load_zephyr_ecg(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load ecg data from zephyr

    :param str data_path: path to data
    :return: zephyr data
    :rtype: np.ndarray
    """
    return parse_zephyr_file(data_path)


def _load_hexoskin(data_path: str) -> Dict[str, List[Union[float, List[float]]]]:
    """ Load data from hexoskin

    :param str data_path: path to data
    :return: hexoskin data
    :rtype: Dict[str, List[Union[float, List[float]]]]
    """
    def check_digit(dd):
        return dd.replace('.', '', 1).replace('\n', '', 1).replace('-', '').replace('e', '', 1).isdigit()
    result_dict = {kk: [] for kk in HEXOSKIN_KEYS}
    with open(data_path) as data:
        for readline in data:
            splitline = readline.split(',')
            if splitline[0] in HEXOSKIN_KEYS:
                result_dict[splitline[0]].append([float(aa) for aa in splitline[1:] if check_digit(aa)])
    return result_dict


def _load_natus(data_path: str) -> Dict[str, Union[int, np.ndarray]]:
    """ Load data from natus

    :param str data_path: path to data
    :return: natus data
    :rtype: np.ndarray
    """
    def renamer(x: str):
        if x.startswith('e_'):
            x = 'e_'.join(x.split('e_')[1:])
        return x.replace('decription', 'description')

    if data_path.endswith('.csv'):
        return pd.read_csv(data_path).rename(renamer, axis='columns')
    return_dict = {}
    raw = mne.io.read_raw_edf(data_path, preload=True)
    return_dict['fs'] = int(raw.info['sfreq'])
    return_dict['data'] = raw[raw.ch_names.index('ECG-RA')][0][0] - raw[raw.ch_names.index('ECG-LA')][0][0]
    return_dict['breathing'] = raw[raw.ch_names.index('Chest')][0][0]
    return return_dict


def _load_gps(data_path: str) -> np.ndarray:
    """ Load speed in knots from gps

    :param str data_path: path to data
    :return: speed from gps
    :rtype: np.ndarray
    """
    def line_count(lines: list) -> int:
        """ Count lines in which the string '$GNRMC' appears

        :param lines: list of lines
        :return: number of lines in which the string '$GNRMC' appears
        :rtype: int
        """
        n = 0
        for row in lines:
            if '$GNRMC' in row:
                n += 1
        return n
    with open(data_path) as f:
        lines_list = f.readlines()
        needed_data = np.zeros(line_count(lines_list))
        cnt = 0
        for line in lines_list:
            try:
                if '$GNRMC' in line:
                    line = line.split(",")
                    needed_data[cnt] = float(line[7])
                    cnt += 1
            except ValueError:
                needed_data[cnt] = 0 if cnt == 0 else needed_data[cnt - 1]
                cnt += 1
    return needed_data * 1.852  # Return speed in knots
