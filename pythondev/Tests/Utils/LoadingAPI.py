import pytz

from Configurations import spot_config
from Tests.Constants import STATUS_TO_CLASS, VITAL_SIGN_LIMITS, DELIVERED
from Tests.SessionHandler import _load
from Tests.Utils.CloudDataUtils import load_PLLConfig_bandwidth
from Tests.Utils.DBUtils import get_reference_systems_list, load_ref_from_npy, find_not_empty_file, \
    load_reference_ie, sort_ref_paths, load_ref, sensor_types, generate_npy_ref_path, shift_reference, \
    add_date_range_index
from Tests.Utils.IOUtils import load, save
from pylibneteera.ExtractPhaseFromCPXAPI import PhaseAPI
from Tests.vsms_db_api import *
from pylibneteera.datatypes import NeteeraSensorType

import datetime
from typing import Union
import numpy as np
import pandas as pd

from pylibneteera.math_utils import normal_round

from Tests.Utils.TestsUtils import last_delivered_version


def _load_reference_stat(setup_id, db, force):
    occ = load_reference(setup_id, 'occupancy', db, force)
    if occ is None:
        return None
    zrr = load_reference(setup_id, 'zrr', db, force)
    motion = load_reference(setup_id, 'motion', db, force)
    if not motion or not zrr:
        return occ
    motion = motion * occ * (1 - zrr)
    return np.array(occ * (zrr + 1) + 2 * motion, dtype=int)


def load_prediction(setup: int, vs: str, db: DB):
    """loads the npy prediction from the latest version release folder if it exists
    :param setup: setup id
    :param vs: vital sign
    :param db: dbv object
    :return the prediction from the latest version release folder if it exists"""

    stats_latest_version = os.path.join(DELIVERED, last_delivered_version(), 'stats')
    for subdir in os.listdir(stats_latest_version):
        for mode in ['dynamic', 'spot']:
            results_dir = os.path.join(stats_latest_version, subdir, mode)
            if os.path.isdir(results_dir) and setup in get_list_of_setups_in_folder_from_vs(results_dir, vs):
                results = load_pred_rel_from_npy(results_dir, setup, vs, False)['pred']
                return add_date_range_index(results, setup, db)
    print(f'could not find prediction setup {setup} vs {vs} in {stats_latest_version}')


def load_nes(setup_id=None, db: DB=None, path=None) -> dict:
    """ Load neteera radar sensor data and information. One should provide path or setup id and db object

    :param setup_id: setup id in db
    :param db: database object
    :param str path: File path to reference data file
    :return: dict of NES sensor data.
        The output['data'] holds the cpx data in a dataframe where each row is marked by the time it was recorded
        (the time is not recorded online by the embedded system an assessment)
        The columns are the different bins that were saved.

    """
    if db is not None:
        npy_path = db.setup_ref_path_npy(setup=setup_id, sensor=Sensor.nes, vs=VS.cpx)
        if npy_path is not None and len(npy_path):
            npy_loaded = np.load(npy_path, allow_pickle=True)
            try:
                return npy_loaded.item()    # works from data from the cloud
            except ValueError:
                if isinstance(npy_loaded[1], dict):
                    loaded_dict = npy_loaded[1]
                    gap = npy_loaded[2]
                    cpx_data = npy_loaded[0]
                    output = {'data': add_date_range_index(cpx_data, setup_id, db, freq=datetime.timedelta(seconds=gap)),
                              'framerate': 1 / gap, 'is_raw': False}
                    output.update(loaded_dict)
                    output.update(db.setup_radar_config(setup_id))
                    return output
                else:
                    raise   # todo
        print('could not load npy cpx data, parsing ttlog')
        if path is None:
            path = db.setup_ref_path(setup_id, Sensor.nes)[0]
    if db is not None and db.mysql_db == 'neteera_cloud_mirror':
        paths = db.setup_ref_path(setup_id, Sensor.nes, search='raw_data')
        if len(paths):
            return load_data_from_json(paths, db, setup_id)
    else:
        output = _load(path, NeteeraSensorType.fmcw)
        if setup_id is not None:
            output.update(db.setup_radar_config(setup_id))
        return output


def load_reference(setup_id: int, vital_sign: Union[str, list], db: DB, force=False, silent=False, use_saved_npy=True):
    """ loads the reference from the DB

    :param int setup_id: setup number
    :param vital_sign: if an str so the output will be an array or pd.Series.
    If it is a list of vital signs the output will be a pd.DataFrame with each vital sign as a column
    :param db:  vsms_db_api holder
    :param bool force: ignore the validity of the reference
    :param bool silent: don't print out issues
    :param bool use_saved_npy: some reference files are stored twice: 1 as the "original" file given by the reference
    system, and another one is a .npy file that is ready for use. The default is using the .npy.
    Making this option False should be done carefully as it can change the .npy saved in the db
    """
    if not isinstance(vital_sign, str):
        return pd.DataFrame({v: load_reference(setup_id, v, db, force, silent, use_saved_npy) for v in vital_sign})
    reference_data = None
    ref_path = None
    try:
        if vital_sign == 'stat':
            return _load_reference_stat(setup_id, db, force)
        elif vital_sign == 'motion':
            rest = load_reference(setup_id, 'rest', db, force)
            return None if rest is None else 1 - rest
        ref = get_reference_systems_list(setup_id, vital_sign, db)
        if ref is None:
            return None
        if not force and ((vital_sign in [str(x) for x in VS] and
                           db.setup_data_validity(setup_id, Sensor[ref.lower()], VS[vital_sign]) == 'Invalid') or
                          db.setup_data_validity(setup_id, Sensor[ref.lower()]) == 'Invalid'):
            if not silent:
                print(f'reference_data for setup {setup_id} with vital sign {vital_sign} is invalid')
            return
        if use_saved_npy:
            try:
                return load_ref_from_npy(setup_id, ref, vital_sign, db)
            except (IndexError, KeyError, TypeError):
                if vital_sign not in ['occupancy', 'zrr', 'speaking', 'motion', 'ie'] and not silent:
                    print(f"can't load from npy ref data. setup: {setup_id} ref: {ref}")

        if vital_sign in ['hri', 'bbi']:
            ref_paths = db.setup_ref_path(setup_id, sensor=Sensor[ref.lower()], search='ECG')
            ref_path = find_not_empty_file(ref_paths)
        elif vital_sign == 'ie':
            ref_path, reference_data = load_reference_ie(setup_id, ref.lower(), db)
        else:
            ref_paths = db.setup_ref_path(setup_id, sensor=Sensor[ref.lower()])
            if len(ref_paths) > 1:
                ref_path = sort_ref_paths (ref_paths, Sensor[ref.lower()], vital_sign)
            else:
                ref_path = ref_paths[0]
            reference_data = load_ref(ref_path, sensor_types[ref], VS(vital_sign))
    # except ZeroDivisionError:
    except (TypeError, IndexError, ValueError, KeyError):
        if not silent:
            print('No reference_data for setup {} with vital sign {}'.format(setup_id, vital_sign))
    #if reference_data is not None and len(reference_data) > 0:
    #    db.insert_npy_ref(setup_id, ref.lower(), vital_sign, generate_npy_ref_path(ref_path, vital_sign), reference_data)
    return reference_data


def load_phase(setup_id, db):
    phase_path = db.setup_ref_path_npy(setup_id, Sensor.nes, VS.phase)
    return None if phase_path is None else load(phase_path)


def load_pred_rel_from_npy(folder, idx, vs, status_to_class=True):
    output = {'rel': np.array([])}
    if os.path.basename(folder) == 'spot':
        output['pred'] = load(os.path.join(folder, f'{idx}_{vs}_spot.data'))['spot_' + vs]
        if vs in ['hr', 'rr']:
            output['pred'] = [np.round(output['pred'])]
    elif vs == 'stat':
        output['pred'] = load(os.path.join(folder, f'{idx}_{vs}.data'))
        if status_to_class:
            output['pred'] = np.array([STATUS_TO_CLASS[p] for p in output['pred']])
    else:
        output['pred'] = [normal_round(x) for x in load(os.path.join(folder, f'{idx}_{vs}.npy'))]
        if vs in ['hr', 'rr']:
            output['rel'] = load(os.path.join(folder, f'{idx}_{vs}_reliability.npy'))
            try:
                output['high_quality'] = load(os.path.join(folder, f'{idx}_{vs}_high_quality_indicator.npy'))
            except FileNotFoundError:
                output['high_quality'] = load(os.path.join(folder, f'{idx}_{vs}_reliability.npy'))
    return output


def load_pred_rel_from_csv(folder, idx, vs):
    file_name = os.path.join(folder, f'{idx}.csv')
    pred_df = pd.read_csv(file_name)
    if f'{vs}Med' in pred_df:
        pred = pred_df[f'{vs}Med'].values
        if pred[0] == -3:
            pred[0] = -1
    elif vs in pred_df:
        pred = pred_df[vs].values
    else:
        return {'pred': np.array([]), 'rel': np.array([])}
    reliability = pred_df[f'{vs}_cvg'].values if f'{vs}_cvg' in pred_df else np.array([])
    high_quality = pred_df.get(f'{vs}_hq', pred_df.get(f'{vs}_cvg'))
    if vs == 'stat':
        pred = np.array([STATUS_TO_CLASS[x.lower().replace('-', ' ')] for x in pred])
    elif vs in ['hr', 'rr']:
        pred = [int(x + 0.5) if x >= 0 else -1 for x in pred]
    return {'rel': reliability, 'pred': pred, 'high_quality': high_quality}


def load_pred_high_qual(folder, idx, vs):
    if os.path.exists(os.path.join(folder, f'{idx}.csv')):
        return load_pred_rel_from_csv(folder, idx, vs)
    else:
        return load_pred_rel_from_npy(folder, idx, vs)


def get_list_of_setups_in_folder_from_vs(folder: str, suffix: str):
    suffix = suffix.replace('.npy', '')
    if suffix.endswith('_'):
        suffix = suffix[:-1]
    if suffix[0] == '_':
        suffix = suffix[1:]

    return dict(sorted([(int(re.findall(r'\d+', file)[0]), file) for file in os.listdir(folder)
                if re.fullmatch(fr'\d+((_{suffix}(_spot)?(\.npy|\.data))|(\.csv))', file)]))


def get_setups_union_folders(folders, vs):
    return set.union(*[set(get_list_of_setups_in_folder_from_vs(fol, vs).keys()) for fol in folders])


def get_setups_intersect_folders(folders, vs):
    return set.intersection(*[set(get_list_of_setups_in_folder_from_vs(fol, vs).keys()) for fol in folders])


def _calc_phase_from_cloud(bin_nums, iq, fs):
    bin_nums.fillna(method='ffill', inplace=True)
    bin_changes = np.argwhere(np.diff(bin_nums))[:, 0] + 1
    current_section_start = 0
    phase_sections = []
    for bin_change in list(bin_changes) + [len(iq)]:
        section = iq[current_section_start * fs: bin_change * fs]
        section_phase = PhaseAPI().iq_to_phase(section, input_fs=fs, use_dll=False)
        phase_sections.append(section_phase)
        current_section_start = bin_change
    return np.concatenate(phase_sections)


def _get_time_index(seconds, periods, gap):
    for i, row in enumerate(seconds):
        if row['last_modified'] != -1:
            str_time = row['last_modified']
            break

    start_datetime = datetime.datetime.strptime(str_time, '%Y-%m-%dT%H:%M:%S') - datetime.timedelta(seconds=i + 1)
    start_datetime = start_datetime.replace(tzinfo=pytz.timezone('utc')).astimezone(pytz.timezone('Asia/Jerusalem'))
    return pd.date_range(start=start_datetime, periods=periods, freq=f'{gap}S')


def load_data_from_json(file_paths, db, setup=None):
    file_path = file_paths[0] if setup is None else [x for x in file_paths if str(setup) in x][0]
    print(f'loading data from {file_path}')
    raw = load(file_path)
    fs = int(raw['session_metadata']['fs'])
    gap = 1 / fs
    seconds = raw['seconds']
    data = [np.array(x['data'].split(), dtype=int) for x in seconds]
    data = [x if len(x) else np.ones(fs * 2) * np.nan for x in data]
    iq = np.dot(
        np.stack(data).reshape(np.size(data)//2, 2), [1, 1j])
    bin_nums = pd.Series([x['bins'] for x in seconds])
    bin_nums[bin_nums == -1] = np.nan
    time_index = _get_time_index(seconds, len(iq), gap)
    iq_time_series = pd.DataFrame(iq, index=time_index)
    parameters = db.setup_radar_config(9700 if setup is None else setup)
    if setup is None:
        setup = db.all_setups()[-1]
    parameters['BW'] = parameters['PLLConfig_bandwidth'] if len(parameters) > 0 else load_PLLConfig_bandwidth(setup, db)
    parameters['dist'] = 0
    parameters['bins'] = [0]
    parameters['framerate'] = fs
    parameters.update({'data': iq_time_series, 'gap': gap, 'cloud_bins': bin_nums})
    db.insert_npy_ref(setup, Sensor.nes, VS.cpx, file_path.replace('cloud_raw_data.json', 'cpx.npy'), parameters)
    db.insert_npy_ref(setup, Sensor.nes, VS.phase, file_path.replace('cloud_raw_data.json', 'phase.npy'),
                      pd.Series(_calc_phase_from_cloud(bin_nums, iq, fs), index=time_index))
    return parameters


def load_missing_indexes(setup_id, db):
    if db.mysql_db == 'neteera_db':
        db.update_mysql_db(setup_id)
    paths = db.setup_ref_path()


def load_shifted_reference(setup_id, vs, db):
    reference = load_reference(setup_id, vs, db, force=True)
    if reference is None:
        return None
    return shift_reference(setup_id, vs, reference, db)


class ReferenceHolder:
    def __init__(self, db, force=False):
        self.db = db
        self.force = force
        self.dynamic = dict()
        self.spot = {'hr': dict(), 'rr': dict()}

    def load_ref(self, setup_nums, vs):
        self.dynamic[vs] = dict()
        self.spot[vs] = dict()
        for num in setup_nums:
            gt = self.load_ground_truth(num, vs)
            if gt is not None:
                self.dynamic[vs][num] = gt
        if vs in self.spot:
            maximum_window_from_end = spot_config['setup']['maximum_window_from_end']
            self.spot[vs] = {num: [np.median(gt[-maximum_window_from_end:])] for num, gt in self.dynamic[vs].items()}

    def load_ground_truth(self, setup_num, vital):
        gt = load_reference(setup_num, vital, self.db, force=self.force)
        if gt is None:
            return None
        if vital not in ['ie', 'stat'] and not self.force and gt is not None and not np.all(gt > 0):
            print(f'setup {setup_num} has values the are not positive in gt. vs {vital}')
            return
        elif vital == 'bbi':
            max_bbi = VITAL_SIGN_LIMITS['bbi']['max']
            if len(gt) >= 2 and np.max(np.diff(gt)) > max_bbi:
                print(f'skipping setup {setup_num} bbi larger than {max_bbi} ms')
                return None
        return gt

    def get_reference(self, vs, num=None, is_spot=False):
        relevant_ref = self.spot[vs] if is_spot and vs in self.spot else self.dynamic[vs]
        if isinstance(num, int):
            return relevant_ref[num]
        if num is None:
            return relevant_ref

    def get_setups_list(self, vs, is_spot=False):
        relevant_ref = self.spot[vs] if is_spot and vs in self.spot else self.dynamic[vs]
        return list(relevant_ref.keys())


if __name__ == "__main__":
    # examples:

    # prediction data
    # print(load_prediction(6284, 'stat', db).iloc[::1000])

    db = DB()
    db.update_mysql_db(9838)

    # # reference data
    print(np.unique(load_reference(9838, 'apnea', db)))

    # phase data
    print(len(load_phase(9838, db)) / 500 / 3600)

    # nes radar cpx data
    print(load_nes(9838, db)['data'].shape)
