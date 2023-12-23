# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
# Utils for database loading etc.
# Apposed to vsms_db_api.py this script was created by the Algo team

import Tests.pathmagic  # noqa

from Tests.ReferenceUtils.IEtool import get_ie_ratio_spirometer
from Tests.Utils.IOUtils import load
from Tests.Utils.TestsUtils import intersect, remove_tails
from Tests.Tools_for_logger.NHW_edf import edf_to_npy
from Tests.ReferenceUtils.data_formatters import *
from Tests.SessionHandler import _load
from Tests.vsms_db_api import *

from pylibneteera.datatypes import SensorType

import csv
import datetime
import numpy as np
from typing import Dict, Tuple, List
import pandas as pd
import pytz

sensor_types = {'BIOPAC': SensorType.biopac, 'SPO2': SensorType.spO2, 'ECG_WAVE_II': SensorType.ecg,
                'ECG_SPO2': SensorType.spO2, 'BITALINO': SensorType.bitalino, 'NATUS': SensorType.natus,
                'DN_ECG': SensorType.dn_ecg, 'HEXOSKIN': SensorType.hexoskin, 'ZEPHYR_BW': SensorType.zephyr_bw,
                'ZEPHYR_ECG': SensorType.zephyr_ecg, 'ZEPHYR_HR_RR': SensorType.zephyr_hr_rr,
                # ZEPHYR is not supported in this way. Please you the same method as ecg for epm_10m
                'GT_CS': SensorType.stat, 'GT_OCCUPANCY': SensorType.stat, 'GT_REST': SensorType.stat,
                'GT_SPEAKING': SensorType.stat, 'GT_STATIONARY': SensorType.stat, 'GT_ZRR': SensorType.stat,
                'ELBIT_ECG_SAMPLE': SensorType.elbit_ecg_sample, 'IMOTION_BIOPAC': SensorType.imotion_biopac,
                'EPM_10M': SensorType.epm_10m, 'CAPNOGRAPH_PC900B': SensorType.capnograph_pc900b,
                'SPIROMETER': SensorType.spirometer}

# also used to map all vs available per sensor in db support
reference_order = {'hr': ['EPM_10M', 'CAPNOGRAPH_PC900B', 'ZEPHYR_HR_RR', 'SPO2', 'ECG_WAVE_II', 'BIOPAC', 'BITALINO',
                          'HEXOSKIN', 'NATUS', 'RESPIRONICS_ALICE6', 'DN_ECG', 'ELBIT_ECG_SAMPLE', 'IMOTION_BIOPAC'],
                   'rr': ['EPM_10M', 'CAPNOGRAPH_PC900B', 'ZEPHYR_HR_RR', 'BIOPAC', 'HEXOSKIN', 'NATUS',
                          'RESPIRONICS_ALICE6', 'ELBIT_ECG_SAMPLE', 'IMOTION_BIOPAC'],
                   'ie': ['SPIROMETER', 'EPM_10M', 'BIOPAC'],
                   'bbi': ['EPM_10M', 'ZEPHYR_ECG', 'DN_ECG', 'BIOPAC'],
                   'bp': ['EPM_10M', 'ELBIT_ECG_SAMPLE'],
                   'stat': ['GT_OCCUPANCY', 'GT_REST', 'GT_SPEAKING', 'GT_STATIONARY', 'GT_ZRR'],
                   'apnea': ['NATUS', 'RESPIRONICS_ALICE6'],
                   'chest': ['NATUS', 'RESPIRONICS_ALICE6'],
                   'nasalpress': ['NATUS', 'RESPIRONICS_ALICE6'],
                   'spo2': ['NATUS', 'RESPIRONICS_ALICE6', 'EPM_10M'],
                   'sleep_stages': ['NATUS', 'RESPIRONICS_ALICE6'],
                   'pleth': ['NATUS', 'RESPIRONICS_ALICE6', 'EPM_10M']}
for signal in ['ecg_la', 'ecg_ra', 'ecg_ll']:
    reference_order[signal] = ['NATUS']


ref_data_fetchers = {VS.hr: {SensorType.spO2: None,
                             SensorType.ecg: ecg_formatter(fs=250, win_sec=10),
                             SensorType.biopac: biopac_hr_formatter(),
                             SensorType.epm_10m: epm_10m_hr_formatter()},
                     VS.rr: {SensorType.biopac: biopac_rr_formatter(),
                             SensorType.elbit_ecg_sample: elbit_ecg_sample_rr_formatter(),
                             SensorType.epm_10m: epm_10m_rr_formatter(),
                             SensorType.capnograph_pc900b: capnograph_pc900b_rr_formatter()},
                     VS.ie: {SensorType.biopac: biopac_inhale_exhale_formatter(win_sec=30),
                             SensorType.epm_10m: epm_10m_ie_formatter,
                             SensorType.spirometer: spirometer_ie_formatter},
                     VS.ra: {},
                     VS.bbi: {SensorType.biopac: biopac_peaks_formatter(),
                              SensorType.epm_10m: epm_10m_peaks_formatter()},
                     VS.bp: {SensorType.elbit_ecg_sample: elbit_ecg_sample_bp_formatter(),
                             SensorType.epm_10m: epm_10m_bp_formatter()},
                     VS.stat: {SensorType.stat: None},
                     VS.occupancy: {SensorType.stat: None},
                     VS.rest: {SensorType.stat: None},
                     VS.zrr: {SensorType.stat: None},
                     VS.posture: {SensorType.natus: natus_posture_formatter},
                     VS.apnea: {SensorType.natus: natus_apnea_formatter},
                     VS.sleep_stages: {SensorType.natus: natus_sleep_stages_formatter}}


def load_ref(path: str, sensor_type: SensorType, vital_sign_type: VS, data: np.ndarray = None) -> np.ndarray:
    """ Loads reference system data

    :param str path: File path to reference data file
    :param SensorType sensor_type: Type of sensor from which the data is collected
    :param VS vital_sign_type: Requested reference data vital sign
    :param np.ndarray data: if data already loaded, no need to re-load, just to format
    :return: An array of ground truth values
    :rtype: np.ndarray
    """
    if data is None:
        data = _load(path, sensor_type)
    if ref_data_fetchers[vital_sign_type][sensor_type] is None:
        return data
    return ref_data_fetchers[vital_sign_type][sensor_type](data)


def get_reference_systems_list(setup: int, vs: str, db):
    """ provide the reference sorted system names by their priority """
    setup_ref_systems = db.setup_ref(setup=setup)
    gt_vs = 'GT_' + vs.replace('motion', 'rest').upper()
    if gt_vs in reference_order['stat']:
        return gt_vs
    else:
        order = reference_order.get(vs, reference_order['hr'])
        for ref in order:
            if ref in setup_ref_systems:
                return ref
    ref_systems_without_nes = [x for x in setup_ref_systems if x != 'NES']
    if len(ref_systems_without_nes) > 1:
        print('WARNING: more than 1 ref system detected!   \n systems:\t', *ref_systems_without_nes)
    return ref_systems_without_nes[0]
    print(f'no relevant reference system found for setup {setup}, vital sign {vs}')


def sort_ref_paths(paths, sensor, vs=None):
    search = '.edf'
    if str(sensor) == 'epm_10m':
        search = 'ParameterData'
    elif str(sensor) == 'natus':
        if vs in ['apnea', 'annotations', 'sleep_stages', 'posture']:
            search = 'annotations'
    for p in paths:
        if search in p:
            return p


def ref_sensor_type(ref):
    return sensor_types[ref]


def find_not_empty_file(paths):
    for path in paths:
        try:
            pd.read_csv(path)
            return path
        except pd.errors.EmptyDataError:
            pass
    raise TypeError


def load_ref_from_npy(setup_id, ref, vital_sign, db):
    vital_sign = vital_sign.replace('bbi', 'hri')
    ref_npy_path = db.setup_ref_path_npy(setup=setup_id, sensor=Sensor[ref.lower()], vs=VS[vital_sign])
    if 'hri' not in ref_npy_path or 'BIOPAC' not in ref_npy_path:
        return load(ref_npy_path)


def load_reference_ie(setup, ref, db):
    if ref.lower() == 'spirometer':
        path = db.setup_ref_path(setup, sensor=Sensor.spirometer)[0]
        spirometer_data = pd.read_csv(path, sep=r'\s+')
        vol = spirometer_data.Vol.values
        time_vec = spirometer_data.Time.values / 1000   # from millisecond to sec
        gap = time_vec[1] - time_vec[0]
        fs = int(1 / gap)
        ref_data = get_ie_ratio_spirometer(vol, fs)
    else:
        path = db.setup_ref_path(setup, sensor=Sensor.epm_10m, search='CO2')[0]
        co2 = pd.read_csv(path, header=None).to_numpy()
        co2 = co2[:, 1:].flatten()
        ref_data = get_ie_rate(co2)
    return path, ref_data


def generate_npy_ref_path(path: str, vs: str):
    index_of_point = path.rfind('.')
    out = path if index_of_point == -1 else path[: index_of_point]
    return out + f'_{vs}.npy'


def shift_reference(setup_id, vs, un_shifted_ref, db):
    shift = calculate_delay(setup_id, vs.replace('stat', 'zrr'), db)
    if isinstance(un_shifted_ref, pd.Series):
        fs = int(1 / un_shifted_ref.index.freq.delta.total_seconds())
        shifted_ref = list(un_shifted_ref)[shift:] if shift >= 0 else [None] * abs(shift) * fs + list(un_shifted_ref)
        return add_date_range_index(shifted_ref, setup_id, db, un_shifted_ref.index.freq)
    else:
        shifted_ref = list(un_shifted_ref)[shift:] if shift >= 0 else [None] * abs(shift) + list(un_shifted_ref)
        return add_date_range_index(shifted_ref, setup_id, db)


def find_back_setup_same_session(setup, db):
    for x in db.setup_multi(setup):
        if db.setup_target(x) == 'back':
            return x


def get_cam_hyperlink(setup, db):
    cam_path = db.setup_ref_path(setup, Sensor.cam, operating_sys='Windows')
    if cam_path is not None and len(cam_path):
        return f'=HYPERLINK(\"{cam_path[0]}\", \"vid_{setup}\")'
    return 'missing'


def delay_over_ts_RD(setup, db):
    delay_dict_sessions = {4816: -13, 4988: -13, 4993: None, 5839: -2, 5840: -2, 5897: -62, 5903: -3, 5907: -2,
                           5913: -3, 5977: -3, 5999: -72, 6004: -72, 6005: -72, 6006: None, 6024: -2, 6028: -2,
                           6037: -79, 6041: -80, 6085: -82}
    session_id = db.session_from_setup(setup)
    return delay_dict_sessions.get(session_id)


def find_ref_dt_biopac(ref_path, nes_dt):
    try:
        clk = np.loadtxt(ref_path, skiprows=37, usecols=0)[0]
    except ValueError:
        clk = np.loadtxt(ref_path, skiprows=38, usecols=0)[0]
    return datetime.datetime(nes_dt.year, nes_dt.month, nes_dt.day, 0, 0, 0) + datetime.timedelta(seconds=clk)


def find_ref_dt_epm_10m(ref_path, setup, db):
    with open(ref_path, 'r') as csvfile:
        csv_dict = [row for row in csv.DictReader(csvfile)]
    assert len(csv_dict) > 0, 'reference file is empty'
    t_df = pd.read_csv(ref_path)
    ecg_time = t_df.loc[0][0]
    naive_datetime = datetime.datetime.strptime(ecg_time, '%Y-%m-%d %H:%M:%S')
    time_zone = db.get_setup_time_zone(setup)
    if time_zone == 'None' or time_zone is None:
        time_zone = 'Asia/Jerusalem'
    pytz_zone = pytz.timezone(time_zone).localize(naive_datetime).tzinfo
    return naive_datetime.replace(tzinfo=pytz_zone)


def find_ref_dt_natus(ref_path, idx, db):
    assert idx in db.setup_by_project(Project.nwh)
    ref_dt = edf_to_npy(ref_path, just_date=True)
    return ref_dt


def calc_shift_seconds(nes_dt, ref_dt, force):
    time_shift = abs(nes_dt - ref_dt)
    if not force:
        assert time_shift.days == 0 and time_shift.seconds < 120, 'more than one 2 minutes between nes and gt times'
    return time_shift.seconds if ref_dt >= nes_dt else -1 * time_shift.seconds


def calculate_delay(setup_id, vs, db: DB, use_saved_in_db=True):
    sensor = get_reference_systems_list(setup_id, vs, db)
    if sensor is None:
        print(f'no sensor found for setup {setup_id} vs {vs}')
        return
    sensor = Sensor[sensor.lower()]
    if sensor == Sensor.spirometer:
        return 0
    db_delay = db.setup_delay(setup_id, sensor=sensor)
    if db_delay is not None and use_saved_in_db:
        if sensor == Sensor.natus:
            delay_from_RD = delay_over_ts_RD(setup_id, db)
            if delay_from_RD is not None:
                return db_delay + delay_over_ts_RD(setup_id, db)
        return db_delay  # todo
    nes_dt = db.setup_ts(setup_id, Sensor.nes)['start']
    ref_path = db.setup_ref_path(setup_id, sensor=sensor)[0]
    if sensor in [Sensor.gt_rest, Sensor.gt_speaking, Sensor.gt_occupancy, Sensor.gt_zrr]:
        ref_dt = datetime.datetime.fromtimestamp(int(ref_path.split('_')[-2]))
    elif sensor == Sensor.biopac:
        ref_dt = find_ref_dt_biopac(ref_path, nes_dt)
    elif sensor == Sensor.epm_10m:
        ref_dt = find_ref_dt_epm_10m(ref_path, setup_id, db)
    elif sensor == Sensor.natus:
        ref_dt = find_ref_dt_natus(ref_path, setup_id, db)
    else:
        raise NotImplementedError('reference is not supported')
    shift_sec = calc_shift_seconds(ref_dt, nes_dt, sensor == Sensor.natus)
    db.set_delay(setup_id, sensor=sensor, value=shift_sec)
    if db.setup_delay(setup_id, sensor) != shift_sec:
        print('delay didnt change!!!!')
    else:
        print(f'delay set setup {setup_id}, sensor {sensor}, shift = {shift_sec}')
    return shift_sec


def match_lists_by_ts(prediction: iter, reference: list, setup_id: int, vs: str, db, nwh_adjust=False)\
        -> Tuple[iter, iter, int]:
    if isinstance(reference, pd.Series):
        return prediction, reference.to_numpy(), 0
    shift = calculate_delay(setup_id, vs, db)
    if nwh_adjust:
        shift += delay_over_ts_RD(setup_id, db)
    if shift >= 0:
        reference = reference[shift:]
    else:
        prediction = prediction[-shift:]

    return *remove_tails(prediction, reference), shift


def setup_types(db) -> Dict[str, List]:
    bed_top = db.setup_by_mount(Mount.bed_top) + db.setup_by_mount(Mount.lab_ceiling)
    bed_bottom = db.setup_by_mount(Mount.bed_bottom)
    bed_side = db.setup_by_mount(Mount.bed_side)
    lab_wall = db.setup_by_mount(Mount.lab_wall)
    seat_back = db.setup_by_mount(Mount.seat_back)
    motion = db.setup_by_state(State.is_motion, True)
    rest = set(db.setup_by_state(State.is_rest, True)) - set(motion)
    back = db.setup_by_target(Target.back)
    front = db.setup_by_target(Target.front)
    side = db.setup_by_target(Target.side)
    shawarma = db.setup_by_target(Target.altered)
    lying = db.setup_by_posture(Posture.lying)
    sitting = db.setup_by_posture(Posture.sitting)
    szmc = set(db.setup_by_project(Project.szmc))
    cen_exel = db.benchmark_setups(Benchmark.cen_exel)
    empty = db.setup_by_state(state=State.is_empty, value=True) + db.setup_by_state(state=State.is_hb, value=True)
    standing = db.setup_by_posture(Posture.standing)
    distance_350 = db.setup_distance_equals(350)
    distance_500 = db.setup_distance_equals(500)
    distance_900 = db.setup_distance_equals(900)
    distance_1000 = db.setup_distance_equals(1000)
    distance_1500 = db.setup_distance_equals(1500)
    distance_1600 = db.setup_distance_equals(1600)
    distance_1800 = db.setup_distance_equals(1800)
    distance_500_1200 = db.setup_distance_in_range(500, 1200)
    distance_1201_1599 = db.setup_distance_in_range(1201, 1599)
    free_standing = db.setup_by_note('free standing')
    leaning = db.setup_by_note('against the wall')
    stress = db.setup_by_note('stress')
    nati = db.setup_by_subject('nati ')
    duration_under_200sec = set(db.setup_by_max_duration(200))

    return {
        'Stress setups Back': intersect([stress, back]),
        'Stress setups Front (radar to Chest)': intersect([stress, front]),

        'Subject\'s name is Nati': nati,

        'SZMC long Chair Back Rest': set(back) & szmc - duration_under_200sec,
        'SZMC long Chair Front Rest (radar to Chest)': set(front) & szmc - duration_under_200sec,
        'SZMC short Chair Back Rest': intersect([back, szmc, duration_under_200sec]),
        'SZMC short Chair Front Rest (radar to Chest)': intersect([front, szmc, duration_under_200sec]),

        'CenExel Chair Front Rest (radar to Chest)': intersect([front, cen_exel, sitting]),
        'CenExel Above Bed lying on back (radar to Chest)': intersect([front, cen_exel, lying]),
        'CenExel Chair Back Rest': intersect([seat_back, cen_exel]),

        #

        'with Empty/ZRR Chair Behind Back': intersect([empty, seat_back]),
        'with Empty/ZRR Chair Front distance 500mm (radar to Chest)':
            intersect([sitting, empty, front, distance_500]),
        'with Empty/ZRR Chair Front distance 1000mm (radar to Chest)':
            intersect([sitting, empty, front, distance_1000]),
        'with Empty/ZRR Chair Front distance 1500mm (radar to Chest)':
            intersect([sitting, empty, front, distance_1500]),
        'with Empty/ZRR Chair Front (radar to Chest) ': intersect([sitting, empty, front]),
        'with Empty/ZRR Under Bed': intersect([empty, bed_bottom]),
        'with Empty/ZRR Above Bed': intersect([empty, bed_top]),
        'with Empty/ZRR Bed Side': intersect([empty, lab_wall + bed_side]),
        'with Empty/ZRR': sorted(empty),

        'Chair Back Rest ': intersect([rest, back, seat_back]),
        'Chair Front Rest distance 500mm (radar to Chest)': intersect([sitting, rest, front, distance_500]),
        'Chair Front Rest distance 1000mm (radar to Chest)': intersect([sitting, rest, front, distance_1000]),
        'Chair Front Rest distance 1500mm (radar to Chest)': intersect([sitting, rest, front, distance_1500]),
        'Chair Front Rest (radar to Chest) ': intersect([sitting, rest, front]),
        'Chair Side Rest (radar to Chest) ': intersect([sitting, rest, side]),

        'Rest Under Bed lying on Back': intersect([bed_bottom, rest, back]),
        'Rest Under Bed lying on Belly (radar to Chest)': intersect([bed_bottom, rest, front]),
        'Rest Under Bed lying on Side': intersect([bed_bottom, rest, side]),

        'Rest Above Bed lying on back distance 1600mm (radar to Chest)':
            intersect([bed_top, rest, front, distance_1600]),
        'Rest Above Bed lying on Belly distance 1600mm (radar to Back)':
            intersect([bed_top, rest, back, distance_1600]),
        'Rest Above Bed lying on Side distance 1600mm': intersect([bed_top, rest, side, distance_1600]),

        'Rest Above Bed lying on back distance 1800mm (radar to Chest)':
            intersect([bed_top, rest, front, distance_1800]),
        'Rest Above Bed lying on Belly distance 1800mm (radar to Back)':
            intersect([bed_top, rest, back, distance_1800]),
        'Rest Above Bed lying on Side distance 1800mm': intersect([bed_top, rest, side, distance_1800]),

        'Rest Above Bed lying on back between 500mm and 1200mm (radar to Chest)':
            intersect([bed_top, rest, front, distance_500_1200]),
        'Rest Above Bed lying on Belly between 500mm and 1200mm (radar to Back)':
            intersect([bed_top, rest, back, distance_500_1200]),
        'Rest Above Bed lying on Side between 500mm and 1200mm': intersect([bed_top, rest, side, distance_500_1200]),

        'Rest Above Bed lying on back between 1201mm and 1599mm (radar to Chest)':
            intersect([bed_top, rest, front, distance_1201_1599]),
        'Rest Above Bed lying on Belly between 1201mm and 1599mm (radar to Back)':
            intersect([bed_top, rest, back, distance_1201_1599]),
        'Rest Above Bed lying on Side between 1201mm and 1599mm': intersect([bed_top, rest, side, distance_1201_1599]),

        'Rest Above Bed lying on back (radar to Chest)': intersect([bed_top, rest, front]),
        'Rest Above Bed lying on Belly (radar to Back)': intersect([bed_top, rest, back]),
        'Rest Above Bed lying on Side': intersect([bed_top, rest, side]),

        'Rest Bed Side lying on back': intersect([lab_wall, rest, lying, front]),
        'Rest Bed Side lying on Belly': intersect([lab_wall, rest, lying, back]),
        'Rest Bed Side lying on Side': intersect([lab_wall, rest, lying, side]),

        'Standing Free distance 900mm': intersect([standing, free_standing, distance_900]),
        'Standing Free distance 1000mm': intersect([standing, free_standing, distance_1000]),
        'Standing against wall distance 900mm': intersect([standing, leaning, distance_900]),
        'Standing against wall distance 1000mm': intersect([standing, leaning, distance_1000]),
        'Standing side (R&D)': intersect([standing, side]),
        'Standing other': sorted(standing),

        'free Motion Altered distance 350 (lying posture changed during session)': intersect(
            [motion, lying, distance_350]),
        'free Motion Altered distance 500 (lying posture changed during session)': intersect(
            [motion, lying, distance_500]),
        'free Motion Altered distance 1600 (lying posture changed during session)': intersect(
            [motion, lying, distance_1600]),
        'free Motion Altered distance 1800 (lying posture changed during session)': intersect(
            [motion, lying, distance_1800]),


        'Motion Chair Back': intersect([motion, back, sitting]),
        'Motion Side sitting': intersect([motion, side, sitting]),
        'Motion Chair Front (radar to Chest)': intersect([motion, front, sitting]),
        'Motion Altered (lying posture changed during setup)': intersect([motion, shawarma]),
        'Motion Back': intersect([motion, back]),
        'Motion Side': intersect([motion, side]),
        'Motion Front': intersect([motion, front]),
        'Motion': motion
    }


def get_not_invalid_setups(db):
    return db.setup_by_data_validity(Validation.confirmed, Sensor.nes) +\
           db.setup_by_data_validity(Validation.valid, Sensor.nes)


def get_bmi(setup, db):
    info = db.setup_subject_details(setup)
    if info['weight'] is not None and info['height'] is not None:
        return info['weight'] / (info['height'] / 100) ** 2
    try:
        return float(re.findall(r'\d+\.\d', info['note'])[0])
    except (IndexError, TypeError):
        return


def add_date_range_index(array, idx, db, freq='s'):
    nes_dt = db.setup_ts(idx, 'nes')['start']
    array = np.array(array)
    if len(array.shape) == 2:
        return pd.DataFrame(array, index=pd.date_range(start=nes_dt, freq=freq, periods=len(array)))
    elif len(array.shape) == 1:
        return pd.Series(array, index=pd.date_range(start=nes_dt, freq=freq, periods=len(array)))
    else:
        raise ValueError('wrong input dimensions.')


if __name__ == '__main__':
    db = DB()
    calculate_delay(5217, 'rr', db, False)
