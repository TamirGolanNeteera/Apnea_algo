# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
import sys
import traceback
from hashlib import sha256
import MySQLdb
import datetime
from enum import Enum
import numpy as np
import os

import pytz
from packaging import version
import platform
import re
from shutil import copyfile
from typing import Optional, List, Dict, Union
import time


PRINT_QUERY = True

MYSQL_USER = 'moshe'
MYSQL_PASSWORD = '1234'
PORT = 3306  # mysql port
HOST = '10.1.1.53'
LINUX_DIR = '/Neteera/DATA'
WINDOWS_DIR = 'S:/'
DIRS = {'Windows': WINDOWS_DIR, 'Linux': LINUX_DIR}

ADDITIONAL_SENSORS = ['biopac_acq', 'ecg_resp', 'spo2_wave', 'cam', 'gps', 'gsensor']
GT_SENSORS = ['gt_rest', 'gt_zrr', 'gt_stationary', 'gt_occupancy', 'gt_cs', 'gt_speaking', 'gt_recorder']
NES_SENSORS = ['nes', 'nes_res']
NONE_CONTINUES_SENSORS = ['elbit_ecg_sample', 'patient_vs', 'valeo_sensor', 'liegehp']

SENSOR_VS = {'BIOPAC': ['hr', 'rr', 'ra', 'hri', 'ie', 'hrv_nni_mean', 'hrv_sdnn', 'hrv_nni_cv', 'hrv_lf_max_power',
                        'hrv_hf_max_power', 'hrv_lf_auc', 'hrv_hf_auc'],
             'BITALINO': ['hr'],
             'CAPNOGRAPH_PC900B': ['hr', 'rr'],
             'DN_ECG': ['hr', 'hri'],
             'ECG_WAVE_II': ['hr'],
             'ELBIT_ECG_SAMPLE': ['hr', 'rr', 'bp'],
             'EPM_10M': ['hr', 'rr', 'hri', 'bp'],
             'GT_OCCUPANCY': ['occupancy'],
             'GT_REST': ['rest'],
             'GT_SPEAKING': ['speaking'],
             'GT_STATIONARY': ['stationary'],
             'GT_ZRR': ['zrr'],
             'HEXOSKIN': ['hr', 'rr'],
             'IMOTION_BIOPAC': ['hr', 'rr'],
             'NATUS': ['hr'],
             'SPO2': ['hr'],
             'ZEPHYR_BW': ['ra'],
             'ZEPHYR_HR_RR': ['hr', 'rr'],
             'ZEPHYR_ECG': ['hri', 'hrv_nni_mean', 'hrv_sdnn', 'hrv_nni_cv', 'hrv_lf_max_power', 'hrv_hf_max_power',
                            'hrv_lf_auc', 'hrv_hf_auc']}


class NamedEnum(Enum):
    def __str__(self) -> str:
        return self.name


class Gender(NamedEnum):
    """ Type of Gender """
    female = 'Female'
    male = 'Male'


class State(NamedEnum):
    """ Type of State """
    is_rest = "is_rest"
    is_empty = "is_empty"
    is_motion = "is_motion"
    is_hb = "is_hb"
    is_occupied = "is_occupied"
    is_stationary = "is_stationary"
    is_engine_on = "is_engine_on"
    is_driving = "is_driving"
    is_driving_idle = "is_driving_idle"
    is_occupancy = "is_occupancy"  # for cpd usage
    is_speaking = "is_speaking"


class Target(NamedEnum):
    """ Type of Target, i.e. impact location in subject's body """
    back = "back"
    bottom = "bottom"
    front = "front"
    top = "top"
    side = "side"
    altered = "altered"


class Mount(NamedEnum):
    """ Type of Mount, i.e. sensor location """
    seat_back = "seat_back"
    seat_bottom = "seat_bottom"
    seat_side = "seat_side"
    seat_headrest = "seat_headrest"
    bed_bottom = "bed_bottom"
    bed_side = "bed_side"
    bed_top = "bed_top"
    vehicle_ceiling = "vehicle_ceiling"
    lab_ceiling = "lab_ceiling"
    lab_other = "lab_other"
    tripod = "tripod"
    lab_wall = "lab_wall"


class Sensor(NamedEnum):
    """ Type of Sensor """
    nes = "nes"
    nes_res = "nes_res"
    biopac = "biopac"
    spo2 = "spo2"
    gsensor = "gsensor"
    gps = "gps"
    gt_occupancy = "gt_occupancy"  # (legacy name: gt_cs)
    gt_zrr = "gt_zrr"
    gt_rest = "gt_rest"
    gt_speaking = "gt_speaking"
    gt_stationary = "gt_stationary"
    gt_recorder = "gt_recorder"
    ecg_wave_ii = "ecg_wave_ii"
    ecg_spo2 = "ecg_spo2"
    bitalino = "bitalino"
    natus = "natus"
    mhe = "mhe"
    dn_ecg = "dn_ecg"
    cam = "cam"
    hexoskin = "hexoskin"
    zephyr_bw = "zephyr_bw"
    zephyr_ecg = "zephyr_ecg"
    zephyr_hr_rr = "zephyr_hr_rr"
    imotion_biopac = "imotion_biopac"
    epm_10m = "epm_10m"
    capnograph_pc900b = "capnograph_pc900b"
    elbit_ecg_sample = "elbit_ecg_sample"
    patient_vs = "patient_vs"
    valeo_sensor = 'valeo_sensor'
    liegehp = 'liegehp'
    spirometer = 'spirometer'
    respironics_alice6 = 'respironics_alice6'


class VS(NamedEnum):
    """ Type of Vital Sign """
    hr = "hr"
    rr = "rr"
    hrv = "hrv"
    ra = "ra"
    hri = "hri"
    bbi = "bbi"
    hrv_sdnn = "hrv_sdnn"
    hrv_nni_mean = "hrv_nni_mean"
    hrv_nni_cv = "hrv_nni_cv"
    hrv_lf_max_power = "hrv_lf_max_power"
    hrv_hf_max_power = "hrv_hf_max_power"
    hrv_lf_auc = "hrv_lf_auc"
    hrv_hf_auc = "hrv_hf_auc"
    inhale_exhale = "inhale_exhale"
    ie = "ie"  # (same as inhale_exhale) TODO: replace inhale_exhale with ie in db
    saturation = "saturation"
    bp = "bp"
    raw = 'raw'  # NES raw data
    cpx = 'cpx'  # fft raw data
    phase = 'phase' # phase data
    # status gt vs params
    rest = "rest"
    zrr = "zrr"
    stationary = "stationary"
    speaking = "speaking"
    occupancy = "occupancy"
    stat = "stat"
    ecg_la = "ecg-la"
    ecg_ra = "ecg-ra"
    ecg_ll = "ecg-ll"
    nasalpress = "nasalpress"
    chest = "chest"
    pleth = "pleth"
    spo2 = "spo2"
    apnea = "apnea"
    sleep_stages = "sleep_stages"
    posture = "posture"


class Benchmark(NamedEnum):
    """ Type of Benchmark """
    simulation_low_hr = 'simulation_low_hr'
    simulation_high_hr = 'simulation_high_hr'
    high_rr_real_sessions = 'high_rr_real_sessions'
    low_rr_real_sessions = 'low_rr_real_sessions'
    low_hr_real_sessions = 'low_hr_real_sessions'
    nwh = 'nwh'  # trials in sleeping lab carried in Northwell Health, US. bed_top only
    ie_bench = 'ie_bench'
    covid_belinson_sample = 'covid_belinson_sample'
    cw = 'cw'  # sub list of for_test_4 of short sessions with low-seed dependency
    fm_cw = 'fm_cw'  # fmcw=1, nes validity 'Invalid'=0
    fmcw = 'fmcw'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw >='0.4.7.5', ref hr=1, ref validity hr 'Invalid'=0,
    # continues ref =1, duration < 480 sec, note not contains 'Tidal'
    fmcw_raw = 'fmcw_raw'  # fmcw=1, nes validity 'Invalid'=0, raw=1
    fmcw_cpx = 'fmcw_cpx'  # fmcw=1, nes validity 'Invalid'=0, cpx=1
    fmcw_raw_front = 'fmcw_raw_front'  # fmcw=1, nes validity 'Invalid'=0, raw=1, target=front
    fmcw_cpx_front = 'fmcw_cpx_front'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, target=front, fw >='0.4.7.5'
    fmcw_raw_back = 'fmcw_raw_back'  # fmcw=1, nes validity 'Invalid'=0, raw=1, target=back
    fmcw_cpx_back = 'fmcw_cpx_back'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, target=back, fw >='0.4.7.5'
    cpp_fmcw = 'cpp_fmcw'  # fmcw, fw >='0.4.7.5', cpx=1, nes validity 'Invalid'=0, target=back|front, EC=0
    status_fmcw_cpx = 'status_fmcw_cpx'  # EC=0&1 or ZRR=0&1, fmcw=1, cpx=1, nes validity 'Invalid'=0, fw >='0.4.7.5'
    status_fmcw = 'status_fmcw'  # EC=0&1 or ZRR=0&1, fmcw=1, cpx=1 | raw =1, nes validity 'Invalid'=0, fw >='0.4.7.5'
    status_cw = 'status_cw'  # EC=0&1 or ZRR=0&1, cw=1, nes validity 'Invalid'=0
    front_cpx_rest = 'front_cpx_rest'
    for_test_fmcw = 'for_test_fmcw'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw >='0.4.7.5', ref hr=1,
    # ref validity hr 'Invalid'=0, continues ref =1
    beat_to_beat_intervals = 'beat_to_beat_intervals'  # ECG recordings and are at least 2 minutes long
    ec_benchmark = 'ec_benchmark'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw>='0.4.7.5' or
    es_benchmark = 'es_benchmark'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw>='0.4.7.5' or
    # '0.1.3.0' <= fw < '0.3.0.0', EC=1 | ZRR = 1, env='lab', fs=500Hz
    fae_rest = 'fae_rest'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw>='0.4.7.5' or '0.1.3.0' <= fw < '0.3.0.0',
    # EC=0, ZRR=0, Motion=0, Driving=0, 80<duration<600 sec, ref validity hr 'Invalid'=0, ref validity rr 'Invalid'=0,
    # note not contains 'Tidal', env='lab', fs=500Hz, reference delay < 60s, ref hr > 0, ref rr > 0, ref=Mindray,
    # company=neteera|szmc|moh, model=myrtus
    fae_special = 'fae_special'     # setups from fae_rest in the past the were excluded: stress, standing
    # ,changing postures during sessions, out of valid range
    fae_rest_back_front = 'fae_rest_back_front'  # fmcw=1, nes validity 'Invalid'=0, cpx=1, fw>='0.4.7.5' or
    # '0.1.3.0' <= fw < '0.3.0.0', EC=0, ZRR=0, Motion=0, Driving=0, 80<duration<600 sec, ref validity hr 'Invalid'=0,
    # ref validity rr 'Invalid'=0, note not contains 'Tidal', env='lab', fs=500Hz, reference delay < 60s,
    # ref hr > 0, ref rr > 0, target=back & mount=seat_back|bed_bottom or
    # target=front & mount=lab_ceiling|tripod|bed_top, ref=Mindray, company=neteera|szmc|moh, model=myrtus
    szmc_clinical_trials = 'szmc_clinical_trials'  # company = 'szmc', ref hr > 0, ref rr > 0
    cen_exel = 'cen_exel'
    mild_motion = 'mild_motion'  # fmcw = 1, nes validity 'Invalid' = 0, cpx = 1, fw >= '0.4.7.5', EC = 0,
    # ZRR=0, Motion=1, Driving=0, duration>45 sec, ref validity hr 'Invalid'=0, ref validity rr 'Invalid'=0,
    # note not contains 'Tidal', standing=0, env='lab', fs=500Hz, reference delay < 60s, scenario contains ’Mild motion’
    med_bridge = 'med_bridge'
    N130P_rest_benchmark = 'N130P_rest_benchmark'  # all valid setups fro N130P
    N130P_ec_benchmark = 'N130P_ec_benchmark'

class Validation(NamedEnum):
    """ Type of Benchmark """
    valid = 'Valid'
    invalid = 'Invalid'
    confirmed = 'Confirmed'


class Project(NamedEnum):
    """ Name of Company Project"""
    elbit = 'elbit'
    neteera = 'neteera'
    nwh = 'nwh'
    valeo_ths = 'valeo_ths'
    hnt = 'hnt'
    gm = 'gm'
    skoda = 'skoda'
    denso = 'denso'
    mobis = 'mobis'
    fresenius = 'fresenius'
    sterasure = 'sterasure'
    higi = 'higi'
    safran = 'safran'
    olabs = 'olabs'  # Cube-AI
    szmc = 'szmc'
    moh = 'moh'
    CenExel = 'CenExel'


class FileType(NamedEnum):
    """ File Type"""
    ttlog = 'ttlog'
    tlog = 'tlog'
    log = 'log'
    cfg = 'cfg'
    csv = 'csv'
    txt = 'txt'


class Model(NamedEnum):
    """Model Type"""
    cherry = 'cherry'
    myrtus = 'myrtus'


class Posture(NamedEnum):
    """ Posture Type"""
    sitting = 'sitting'
    lying = 'lying'
    standing = 'standing'


class Environment(NamedEnum):
    """ Type of Environment """
    lab = 'lab'
    vehicle = 'vehicle'


def get_db_path(full_path):
    """the path saved in the db starts from the year number - 4 digits"""
    match = re.search(r'\d{4}([\\/])', full_path)
    if match is not None:
        return full_path[match.regs[0][0]:]


class DB:
    def connect(self):
        for i in range(100):    # Trying to open connection
            try:
                self.conn = MySQLdb.connect(
                    host=HOST, port=PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db=self.mysql_db)
                self.cursor = self.conn.cursor()
                break
            except (MySQLdb.Error, MySQLdb.Warning) as str_err:
                print(str_err)
                time.sleep(1)
                print('Retrying to connect...')

    def __init__(self, db_name='neteera_db'):
        """
        Opens a connection to MySQL DB and sets a new cursor member object
        """
        self.mysql_db = db_name
        self.conn = None
        self.cursor = None
        self.connect()

    def _reopen_if_closed(self):
        if not self.conn.open:
            self.connect()
            return
        try:
            self.conn.ping(False)
        except (MySQLdb.Error, MySQLdb.Warning):
            self.connect()

    def update_mysql_db(self, setup):
        if setup not in self.all_setups():
            self.mysql_db = 'neteera_cloud_mirror' if self.mysql_db == 'neteera_db' else 'neteera_db'
            self.connect()
            print(f'connected to {self.mysql_db}')

    def _execute(self, *args):
        self._reopen_if_closed()
        self.conn.rollback()
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace() is not None:
            for call in traceback.extract_stack():
                if call.filename == __file__ and call.name == '<module>':
                    print(*args, sep='\n')
                    break
        return self.cursor.execute(*args)

    def _execute_and_fetch(self, query, return_size=1):
        try:
            self._execute(query)
            fetched = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                if len(fetched) == 1:
                    result = fetched[0]
                    if len(result) == 1:
                        return result[0]
                    else:
                        return result
                else:
                    if len(fetched[0]) == 1:
                        return [r[0] for r in fetched]
                    else:
                        return fetched
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            print(query)
        if return_size == 1:
            return None
        return [None] * return_size

    def _execute_and_commit(self, query):
        try:
            affected_count = self._execute(query)
            self.conn.commit()
            return affected_count
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            print(query)
            return False

    def setup_fw_smaller_than(self, fw: str) -> Optional[List[int]]:
        """ retrieve setup ids for which firmware is smaller than <fs>
            :param str fw: firmware version
            :return: List of setup ids
            :rtype: List[int]
            """
        vv = fw.split('.')
        if len(vv) != 4:  # version digit numbers e.g. '0.0.0.1'
            return None
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param, nes_param WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'infoFW_version' AND 
            CONCAT(
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 1), '.', -1), 10, '0'), 
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 2), '.', -1), 10, '0'), 
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 3), '.', -1), 10, '0'),
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 4), '.', -1), 10, '0')
            ) < CONCAT(LPAD(%s, 10, '0'), LPAD(%s, 10, '0'), LPAD(%s, 10, '0'), LPAD(%s, 10, '0')) 
            ORDER BY setup.setup_id""", [vv[0], vv[1], vv[2], vv[3]])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_fw_greater_than(self, fw: str) -> Optional[List[int]]:
        """ retrieve setup ids for which firmware is greater than <fs>
            :param str fw: firmware version
            :return: List of setup ids
            :rtype: List[int]
            """
        vv = fw.split('.')
        if len(vv) != 4:  # version digit numbers e.g. '0.0.0.1'
            return None
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param, nes_param WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'infoFW_version' AND 
            CONCAT(
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 1), '.', -1), 10, '0'), 
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 2), '.', -1), 10, '0'), 
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 3), '.', -1), 10, '0'),
            LPAD(SUBSTRING_INDEX(SUBSTRING_INDEX(join_nes_config_param.value, '.', 4), '.', -1), 10, '0')
            ) > CONCAT(LPAD(%s, 10, '0'), LPAD(%s, 10, '0'), LPAD(%s, 10, '0'), LPAD(%s, 10, '0')) 
            ORDER BY setup.setup_id""", [vv[0], vv[1], vv[2], vv[3]])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_fw_equals(self, fw: str) -> Optional[List[int]]:
        """ retrieve setup ids by firmware

        :return: List of setup ids for a given fw version
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param, nes_param WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'infoFW_version' AND 
            join_nes_config_param.value = %s ORDER BY setup.setup_id""", [fw])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_fw(self, setup: int) -> Optional[str]:
        """ retrieve setup firmware

        :return: setup fw version
        :rtype: str
        """
        try:

            self._execute("""SELECT join_nes_config_param.value FROM setup, nes_config, join_nes_config_param, 
            nes_param WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'infoFW_version' AND 
            setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def agc_setups(self) -> Optional[List[int]]:
        """ retrieve setup ids for which AGC control is enabled

        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param, nes_param WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND 
            nes_param.name = 'AGCControl_agcEnabled' AND (join_nes_config_param.value = 'enabled' OR 
            join_nes_config_param.value = 1) ORDER BY setup.setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def is_VSM(self, setup: int) -> Optional[bool]:
        """Check if setup_id is of type 'vsm'('VS-AUTO or 'VS-HC')

        :param int setup: Unique identifier for the setup
        :return: True/False
        :rtype: bool
        """
        try:

            self._execute("""SELECT setup.* FROM setup, setup_type WHERE setup.setup_id = %s AND 
            setup.fk_setup_type_id = setup_type.setup_type_id AND 
            (setup_type.name='VS-HC' or setup_type.name='VS-AUTO')""", [setup])
            count = self.cursor.rowcount
            return True if count > 0 else False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_ext_proc_ref(self, setup: int) -> Optional[List[str]]:
        """ Return List of external process reference sensors for setup
            :param int setup: Unique identifier for the setup
            :return: List of external process reference sensors
            :rtype: List
        """
        sensors = self.setup_ref(setup)
        if sensors:
            return [s for s in sensors if s.lower() not in ADDITIONAL_SENSORS]
        return sensors

    def setup_ref(self, setup: int) -> Optional[List[str]]:
        """ Return List of reference sensors for setup
            :param int setup: Unique identifier for the setup
            :return: List of reference sensors
            :rtype: List
        """
        return self._execute_and_fetch(f"""
        SELECT DISTINCT sensor.name
        FROM setup, data, sensor
        WHERE setup.fk_session_id = data.fk_session_id
        AND data.fk_sensor_id = sensor.sensor_id 
        AND setup.setup_id = {setup}""")

    def setup_duration(self, setup: int) -> Optional[int]:
        """ Return duration of setup
            :param int setup: Unique identifier for the setup
            :return: duration of setup
            :rtype: int
        """
        return self._execute_and_fetch(f"""
            SELECT duration_sec
            FROM setup
            WHERE setup_id = {setup}""")

    def setup_by_max_duration(self, max_duration: int) -> Optional[List[int]]:
        """
        :type max_duration: maximum duration of setup in seconds
        """
        return self._execute_and_fetch(f"""
            SELECT setup_id
            FROM setup
            WHERE duration_sec <= {max_duration}""")

    def setup_target(self, setup: int) -> Optional[str]:
        """ Return setup Target
            :param int setup: Unique identifier for the setup
            :return: setup Target
            :rtype: int
        """
        try:

            self._execute("""SELECT nes_subject_position.name FROM nes_subject_position, setup WHERE 
                    setup.fk_nes_subject_position_uuid = nes_subject_position.nes_subject_position_uuid AND 
                    setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_mount(self, setup: int) -> Optional[str]:
        """ Return setup mount
            :param int setup: Unique identifier for the setup
            :return: setup mount
            :rtype: int
        """
        try:

            self._execute("""SELECT nes_mount_area.name FROM nes_mount_area, setup WHERE 
                    setup.fk_nes_mount_area_uuid = nes_mount_area.nes_mount_area_uuid AND 
                    setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def nes_model_type_id(self, setup_id: int) -> Optional[int]:
        """ Return nes model type of a setup
            :param int setup_id: Unique identifier for the setup
            :return: nes model type of a setup
            :rtype: int
        """
        try:

            self._execute("""SELECT nes_model.fk_nes_model_type_id FROM nes_model, nes_config, setup 
                    WHERE setup.fk_nes_config_id = nes_config.nes_config_id
                    AND nes_config.fk_nes_model_id = nes_model.nes_model_id AND setup.setup_id = %s""", [setup_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def ir_setups(self) -> list:
        """ A list of interferometer setup ids

        :return: A list of interferometer setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, nes_model WHERE 
            setup.fk_nes_config_id = nes_config.nes_config_id AND nes_config.fk_nes_model_id = nes_model.nes_model_id
            AND nes_model.fk_nes_model_type_id = %s ORDER BY setup.setup_id""", [1])  # 1 represents interferometer
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def sr_setups(self) -> list:
        """ A list of radar setup ids

        :return: A list of radar setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, nes_model WHERE 
                    setup.fk_nes_config_id = nes_config.nes_config_id
                    AND nes_config.fk_nes_model_id = nes_model.nes_model_id AND 
                    nes_model.fk_nes_model_type_id = %s ORDER BY setup.setup_id""", [2])  # 2 represents radar
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_distance(self, setup: int) -> Optional[str]:
        """ Get device <radar> distance from target in mm units for a given setup id

        :param int setup: Unique identifier for the setup
        :return: distance in mm
        :rtype: int
        """
        try:
            self._execute("""SELECT target_distance_mm FROM setup WHERE setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_mode(self, setup: int) -> Optional[str]:
        """ Get radar configuration mode by setup

        :param int setup: Unique identifier for the setup
        :return: radar mode <'CW', 'FMCW'>
        :rtype: str
        """
        try:

            self._execute("""SELECT join_nes_config_param.value FROM setup, nes_config, join_nes_config_param, 
            nes_param WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND 
            nes_param.name = 'systemConfig_mode' AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def sr_cw_setups(self) -> Optional[List[int]]:
        """ A list of radar setup ids

        :return: A list of radar setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param, nes_param WHERE 
                                      setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
                                      nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
                                      join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND 
                                      nes_param.name = 'systemConfig_mode' AND join_nes_config_param.value = 'CW' 
                                      ORDER BY setup.setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def sr_fmcw_setups(self) -> list:
        """ A list of radar setup ids

        :return: A list of radar setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, setup_type, nes_config, join_nes_config_param, nes_param 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'systemConfig_mode' 
            AND join_nes_config_param.value = 'FMCW' AND setup.fk_setup_type_id = setup_type.setup_type_id AND 
            (setup_type.name='VS-HC' OR setup_type.name='VS-AUTO' OR setup_type.name='R&D') ORDER BY setup.setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_fs(self, setup: int) -> Optional[float]:
        """ Return sampling frequency of setup

            :param int setup: Unique identifier for the setup
            :return: sampling frequency of setup
            :rtype: int
            """
        out =  self._execute_and_fetch(f"""
        SELECT join_nes_config_param.value
        FROM setup, nes_config, join_nes_config_param,  nes_param
        WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid
        AND  nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid
        AND  join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid
        AND nes_param.name = 'infoFW_framerate'  
        AND setup.setup_id = {setup}""")
        if out is not None:
            return float(out)

    def setup_subject(self, setup: int) -> Optional[str]:
        """ Return subject name of a setup

            :param int setup: Unique identifier for the setup
            :return: subject name
            :rtype: str
            """
        try:

            self._execute("""SELECT subject.name FROM setup, session, subject WHERE 
              setup.fk_session_uuid = session.session_uuid AND session.fk_subject_uuid = subject.subject_uuid AND 
              setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_operator(self, setup: int) -> Optional[str]:
        """ Return operator name of a setup

            :param int setup: Unique identifier for the setup
            :return: operator name
            :rtype: str
            """
        try:

            self._execute("""SELECT op.name FROM setup, session, subject, neteera_staff_operator, subject op 
            WHERE setup.fk_session_uuid = session.session_uuid AND 
            session.fk_neteera_staff_operator_uuid = neteera_staff_operator.neteera_staff_operator_uuid AND 
            neteera_staff_operator.fk_subject_uuid = op.subject_uuid AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __session_date(self, session_id: int) -> Optional[str]:
        """ Retrieve date of session

           :param int session_id: Unique identifier for the session
           :return: date of session
           :rtype: str
           """
        try:

            self._execute("""SELECT date FROM session WHERE session_id = %s""", [session_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __subject_id(self, setup_id: int) -> Optional[int]:
        """ Retrieve the subject_id from setup_id

        :param int setup_id: Unique identifier for the setup
        :return: subject_id: Unique identifier for the subject
        :rtype: str
        """
        try:

            self._execute("""SELECT subject.subject_id FROM setup, session, subject WHERE 
            setup.fk_session_id = session.session_id AND session.fk_subject_id = subject.subject_id AND 
            setup_id = %s""", [setup_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __session_id(self, setup_id: int) -> Optional[int]:
        """ Retrieve the session_id from setup_id

        :param int setup_id: Unique identifier for the setup
        :return: session_id: Unique identifier for the session
        :rtype: str
        """
        try:

            self._execute("""SELECT fk_session_id FROM setup WHERE setup_id = %s""", [setup_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_dir(self, setup: int) -> Optional[str]:
        """ Retrieve path to setup data directory

        :param int setup: Unique identifier for the setup
        :return: path to setup data directory
        :rtype: str
        """
        path = None
        session_id = self.__session_id(setup)
        if session_id is not None:
            dt = datetime.datetime.strptime(str(self.__session_date(session_id)), '%Y-%m-%d %H:%M:%S')
            year, month, day = map(int, str(dt.date()).split('-'))
            data_dir = DIRS[platform.system()]
            path = os.sep.join([data_dir, str(year), str(month), str(day), str(session_id), str(setup)])
        return path

    def setup_delay(self, setup: int, sensor: Sensor, search: Optional[str] = None) -> Optional[int]:
        """ Retrieve delay between nes and reference

        :param int setup: Unique identifier for the setup
        :param int sensor: Name of sensor
        :return: delay between nes and reference
        :param str search: Sensor file name to search
        :rtype: int
        """
        try:
            query = f"""
            SELECT join_setup_data_param.value
            FROM join_setup_data_param, setup, data, sensor, feature_param
            WHERE join_setup_data_param.fk_setup_id = setup.setup_id
            AND join_setup_data_param.fk_data_uuid = data.data_uuid
            AND join_setup_data_param.fk_feature_param_uuid = feature_param.feature_param_uuid
            AND data.fk_sensor_uuid = sensor.sensor_uuid
            AND feature_param.name = 'nes_ref_delay' AND setup.setup_id = {setup} 
            AND sensor.name = '{sensor}'"""
            self._execute(query)
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return int(res[0][0])
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_ref_path(self, setup: int, sensor: Sensor, extension: Optional[FileType] = None,
                       search: Optional[str] = None, operating_sys: Optional[str] = None) -> Optional[List[str]]:
        """ File path for output from a requested sensor for a requested setup

        :param operating_sys:
        :param int setup: Unique identifier for the setup
        :param Sensor sensor: The type of sensor for which the file is being requested
        :param FileType extension: The type of sensor file
        :param str search: Sensor file name to search
        :return: File path for the sensor output
        :rtype: str
        """
        if str(sensor) == 'nes_res':
            return self.__setup_test_ref_path(setup, extension, search)
        else:
            return self.__setup_data_ref_path(setup, sensor, extension, search, operating_sys)

    def __setup_data_ref_path(self, setup_id: int, sensor: Sensor, ext: Optional[FileType] = None,
                              search: Optional[str] = None, operating_system=None) -> Optional[List[str]]:
        """ File path for output from a requested sensor for a requested setup

        :param int setup_id: Unique identifier for the setup
        :param str sensor: The type of sensor for which the file is being requested
        :param FileType ext: The type of sensor file
        :param str search: Sensor file name to search
        :return: File path for the sensor output
        :rtype: str
        """
        results = self._execute_and_fetch(f"""
        SELECT DISTINCT T.DATA_FPATH FROM((SELECT data.fpath AS DATA_FPATH FROM data, sensor WHERE 
                                   data.fk_setup_id = {setup_id} AND data.fk_sensor_id = sensor.sensor_id
                                   AND sensor.name = '{sensor}') 
                                   UNION ALL (SELECT data.fpath AS DATA_FPATH FROM data, sensor, setup, session
                                   WHERE setup.fk_session_id = session.session_id
                                   AND setup.fk_session_id = data.fk_session_id
                                   AND data.fk_sensor_id = sensor.sensor_id AND
                                   setup.setup_id = {setup_id}
                                   AND  sensor.name = '{sensor}'))T ORDER BY LENGTH(T.DATA_FPATH)""")
        path = []
        if results is not None:
            if operating_system is None:
                operating_system = platform.system()
            if isinstance(results, str):
                results = [results]
            for result in results:
                path.append(os.path.join(DIRS[operating_system], result).replace('\\', '/'))
            if not ext and not search:
                if str(sensor) == 'epm_10m':
                    path = [p for p in path if 'ParameterData' in p]
                if str(sensor) == 'nes':
                    path = [p for p in path if p.endswith('tlog') or p.endswith('cloud_raw_data.json')]
            if ext:
                path = [p for p in path if os.path.splitext(p)[1].replace('.', '') == str(ext)]
            if search:
                path = [p for p in path if search in p]
        return path

    def __setup_test_ref_path(self, setup_id: int, ext: Optional[FileType] = None, search: Optional[str] = None) \
            -> Optional[List[str]]:
        """ File path for output from a requested sensor for a requested setup

        :param int setup_id: Unique identifier for the setup
        :param FileType ext: The type of sensor file
        :return: File path for the sensor output
        :param str search: Sensor file name to search
        :rtype: str
        """
        self._execute("""SELECT res_fpath FROM test WHERE fk_setup_id = %s ORDER BY LENGTH(res_fpath)""",
                      [setup_id])
        res = self.cursor.fetchall()
        count = self.cursor.rowcount
        data = [row[0] for row in res]
        path = []
        if count > 0:
            if platform.system() == "Linux":
                for dd in data:
                    path.append(os.path.join(LINUX_DIR, dd).replace('\\', '/'))
            elif platform.system() == "Windows":
                for dd in data:
                    path.append(os.path.join(WINDOWS_DIR, dd).replace('\\', '/'))
        if not ext and not search:
            path = [p for p in path if os.path.splitext(p)[1] in ['.csv']]
        if ext:
            path = [p for p in path if os.path.splitext(p)[1].replace('.', '') == str(ext)]
        if search:
            path = [p for p in path if search in p]
        return path

    def setup_ref_path_npy(self, setup: int, sensor: Sensor, vs: VS = None) -> Optional[str]:
        """ File npy path for output from a requested sensor, setup and vs

        :param int setup: Unique identifier for the setup
        :param str sensor: The type of sensor for which the file is being requested
        :param VS vs: The type of vs
        :return: File path for the sensor output
        :rtype: str
        """
        if str(sensor) == 'nes_res':
            return self.__setup_test_ref_path_npy(setup, vs)
        if str(sensor) == 'nes':
            return self.__setup_nes_path_npy(setup, vs)
        else:
            return self.__setup_data_ref_path_npy(setup, sensor, vs)

    def __setup_data_ref_path_npy(self, setup_id: int, sensor: Sensor, vs: VS = None, operating_system=None) \
            -> Optional[str]:
        """ File npy path for output from a requested sensor, setup and vs

        :param int setup_id: Unique identifier for the setup
        :param str sensor: The type of sensor for which the file is being requested
        :param VS vs: The type of vs
        :return: File path for the sensor output
        :rtype: str
        """
        query = f"""
            SELECT data_npy.fpath
            FROM data_npy
            INNER JOIN data ON data_npy.fk_data_id = data.data_id
            INNER JOIN feature_param ON data_npy.fk_feature_param_id = feature_param.feature_param_id
            INNER JOIN sensor ON data.fk_sensor_id = sensor.sensor_id
            INNER JOIN setup ON  data.fk_session_id = setup.fk_session_id
            WHERE sensor.name = '{sensor}'
            AND setup.setup_id = {setup_id}
            """
        if vs is not None:
            query += f""" AND feature_param.name = '{vs.value}'"""
        res = self._execute_and_fetch(query)
        if res is not None:
            if operating_system is None:
                operating_system = platform.system()
            if isinstance(res, str):
                return os.path.join(DIRS[operating_system], res).replace('\\', '/')
            else:
                return [os.path.join(DIRS[operating_system], x).replace('\\', '/') for x in res]

    def __setup_nes_path_npy(self, setup_id: int, vs: VS, operating_system=None) \
            -> Optional[str]:
        """ File npy path for output from a requested sensor, setup and vs

        :param int setup_id: Unique identifier for the setup
        :param VS vs: The type of vs
        :return: File path for the sensor output
        :rtype: str
        """
        res = self._execute_and_fetch(f"""
            SELECT data_npy.fpath
            FROM data_npy
            INNER JOIN data ON data_npy.fk_data_id = data.data_id
            INNER JOIN feature_param ON data_npy.fk_feature_param_id = feature_param.feature_param_id
            INNER JOIN sensor ON data.fk_sensor_id = sensor.sensor_id
            INNER JOIN setup ON  data.fk_setup_id = setup.setup_id
            WHERE sensor.name = 'nes'
            AND feature_param.name = '{vs}'
            AND setup.setup_id = {setup_id};
            """)
        if res is not None:
            if operating_system is None:
                operating_system = platform.system()
            return os.path.join(DIRS[operating_system], res).replace('\\', '/')

    def __setup_test_ref_path_npy(self, setup_id: int, vs: VS) -> Optional[List[str]]:
        """ File npy path for output from a requested setup and vs

        :param int setup_id: Unique identifier for the setup
        :param VS vs: The type of vs
        :return: File path for the sensor output
        :rtype: str
        """
        try:

            self._execute("""SELECT test_npy.fpath FROM test, test_npy, feature_param WHERE 
            test.test_id = test_npy.fk_test_id AND feature_param.feature_param_id = test_npy.fk_feature_param_id AND 
            test.res_fpath LIKE %s AND test.fk_setup_id = %s AND feature_param.name = %s""",
                          ['%results%', setup_id, vs])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            pth = []
            if count > 0:
                if platform.system() == "Linux":  # Linux
                    pth.append(os.path.join(LINUX_DIR, res[0][0]).replace('\\', '/'))
                elif platform.system() == "Windows":  # Windows
                    pth.append(os.path.join(WINDOWS_DIR, res[0][0]).replace('\\', '/'))
            return pth
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_sensor(self, sensor: Sensor) -> list:
        """ A list of setup ids for which sensor exist

        :param str sensor: Unique identifier name for sensor
        :return: A list of setup ids for which sensor exist
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT DISTINCT T.SETUP_ID FROM((SELECT setup.setup_id AS SETUP_ID FROM setup, data, sensor 
                                      WHERE setup.setup_id = data.fk_setup_id AND data.fk_sensor_id = sensor.sensor_id
                                      AND sensor.name = %s) UNION ALL (
                                      SELECT setup.setup_id AS SETUP_ID FROM setup, session, data, sensor
                                      WHERE setup.fk_session_id = session.session_id
                                      AND setup.fk_session_id = data.fk_session_id
                                      AND data.fk_sensor_id = sensor.sensor_id AND sensor.name = %s))T
                                      ORDER BY T.SETUP_ID""",
                          [sensor, sensor])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def all_setups(self) -> object:
        """ A list of all setup ids

        :return: A list of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup_id FROM setup ORDER BY setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def benchmark_setups(self, benchmark: Benchmark) -> list:
        """ A list of setup ids for which the benchmark column ('for_test'/ 'jenkins') is 'TRUE'
        :param benchmark: benchmark type (e.g: 'for_test')
        :return: A list of setup ids for which the 'for_test' column is 'TRUE'
        :rtype: List[int]
        """
        try:
            self._execute(f"""
            SELECT join_setup_feature_param.fk_setup_id
            FROM join_setup_feature_param, feature_param, feature_param_type
            WHERE join_setup_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid
            AND  feature_param.fk_feature_param_type_uuid = feature_param_type.feature_param_type_uuid
            AND feature_param_type.name = 'benchmark'
            AND feature_param.name = '{benchmark}'
            AND join_setup_feature_param.value = 'TRUE'
            ORDER BY join_setup_feature_param.fk_setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_project_id(self, setup: int, prj: Project) -> Optional[str]:
        """ Return project setup id given by the company project for a given setup id and project name

        :param setup: setup id
        :param Project prj: project name
        :return: project setup id
        :rtype: str
        """
        try:

            self._execute("""SELECT join_setup_feature_param.value FROM join_setup_feature_param, feature_param, 
                    feature_param_type WHERE 
                    join_setup_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
                    feature_param.fk_feature_param_type_uuid = feature_param_type.feature_param_type_uuid AND 
                    feature_param_type.name = 'customer_id' AND join_setup_feature_param.fk_setup_id = %s AND 
                    feature_param.name = %s""", [setup, 'id_{}'.format(prj)])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __session_uuid(self, setup_id: int) -> Optional[str]:
        """ Retrieve session UUID for which the setup id exist

            :param int setup_id: Unique identifier for the setup
            :return: session UUID
            :rtype: str
        """
        try:

            self._execute("""SELECT session_uuid FROM session, setup WHERE 
            setup.fk_session_uuid = session.session_uuid AND setup.setup_id = %s""", [setup_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def set_state(self, setup: int, state: State, value: bool) -> bool:
        """ Set state value for which setup id and state name exists

        :param int setup: Unique identifier for the setup
        :param State state: state name
        :param bool value: bool
        :return: False/True
        :rtype: bool
        """
        param_uuid = self._feature_param_ids(str(state))[0]
        session_uuid = self.__session_uuid(setup)
        if param_uuid and session_uuid:
            return self.__update_state(session_uuid, param_uuid, value)
        return False

    def __update_state(self, session: str, param: str, value: bool) -> bool:
        """ Update state value for which setup id and session UUID exists
           :param str session:  session UUID
           :param str param: state param UUID
           :param bool value: False/True
           :return: False/True
           :rtype: bool
        """
        try:
            affected_count = self._execute("""UPDATE join_session_feature_param SET value = %s 
            WHERE fk_feature_param_uuid = %s AND fk_session_uuid = %s""", [str(value).upper(), param, session])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def setup_by_state(self, state: State, value: bool) -> Optional[List[int]]:
        """ A list of setup ids for which the 'state_name' exist

        :param State state: Unique identifier for the state_name column
        :param bool value: False/True
        :return: A list of setup ids for which state is 'TRUE' OR 'FALSE'
        :rtype: List[int]
        """
        try:
            self._execute("""
            SELECT setup.setup_id
            FROM setup, session, join_session_feature_param, feature_param
            WHERE setup.fk_session_id = session.session_id
            AND session.session_id = join_session_feature_param.fk_session_id
            AND join_session_feature_param.fk_feature_param_id = feature_param.feature_param_id
            AND feature_param.name = %s
            AND join_session_feature_param.value = %s
            """, [state, str(value)])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_target(self, target: Target) -> Optional[List[int]]:
        """ A list of setup ids for which the 'target' exist

        :param str target: Unique identifier for the 'nes_subject_position' column
        :return: A list of setup ids for which target exist
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_subject_position WHERE
                    setup.fk_nes_subject_position_id = nes_subject_position.nes_subject_position_id AND
                    nes_subject_position.name = %s ORDER BY setup.setup_id""", [target])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_mount(self, mount: Mount) -> list:
        """ A list of setup ids for which the 'mount' exist

        :param str mount: Unique name for the Mount
        :return: A list of setup ids for which mount exist
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_mount_area WHERE
                    setup.fk_nes_mount_area_id = nes_mount_area.nes_mount_area_id AND
                    nes_mount_area.name = %s ORDER BY setup.setup_id""", [mount])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_by_posture(self, posture: Posture) -> Optional[list]:
        """ A list of setup ids for which the 'posture' exist

        :param Posture posture: Unique name for the Posture
        :return: A list of setup ids for which posture exist
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, session, subject_posture WHERE 
            setup.fk_session_uuid = session.session_uuid AND 
            session.fk_subject_posture_uuid = subject_posture.subject_posture_uuid AND 
            subject_posture.name = %s ORDER BY setup.setup_id""", [posture])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_environment(self, environment: Environment) -> Optional[list]:
        """ A list of setup ids for which the 'environment' exist

        :param Environment environment: Unique name for the Environment
        :return: A list of setup ids for which posture exist
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, session, environment, environment_type WHERE 
            setup.fk_session_uuid = session.session_uuid AND 
            session.fk_environment_uuid = environment.environment_uuid AND 
            environment.fk_environment_type_uuid = environment_type.environment_type_uuid AND environment_type.name = %s 
            ORDER BY setup.setup_id""", [environment])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __feature_param_id(self, param_name: str) -> Optional[int]:
        """ Retrieve param id for which the 'param_name' exist

                    :param str param_name: Unique identifier for the param_name
                    :return: param_id for which param_name exist
                    :rtype: int
                    """
        try:

            self._execute("""SELECT feature_param_id FROM feature_param WHERE name = %s""", [param_name])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_vs_greater_than(self, param_id: int, min_value: int) -> Optional[list]:
        """ A list of setup ids for which the param value is greater than the value passed

                    :param int param_id: Unique identifier for the param
                    :param int min_value: value of param
                    :return: A list of setup ids for which the constraint: param value grater than 'min_value' exists
                    :rtype: List
                    """
        try:

            self._execute("""SELECT DISTINCT setup.setup_id FROM reference_data_hist, data, setup WHERE 
            data.fk_session_id = setup.fk_session_id AND reference_data_hist.fk_data_id = data.data_id AND 
            reference_data_hist.fk_feature_param_id = %s AND reference_data_hist.bin > %s ORDER BY setup.setup_id""",
                          [param_id, str(min_value)])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_vs_smaller_than(self, param_id: int, max_value: int) -> Optional[list]:
        """ A list of setup ids for which the param value is smaller than the value passed

                    :param int param_id: Unique identifier for the param
                    :param int max_value: value of param
                    :return: A list of setup ids for which the constraint: param value smaller than 'min_value' exists
                    :rtype: List[int]
                    """
        try:

            self._execute("""SELECT DISTINCT setup.setup_id FROM reference_data_hist, data, setup WHERE 
            data.fk_session_id = setup.fk_session_id AND reference_data_hist.fk_data_id = data.data_id AND 
            fk_feature_param_id = %s AND bin < %s ORDER BY setup.setup_id""", [param_id, str(max_value)])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_vs_equals(self, param_id: int, value: int) -> Optional[list]:
        """ A list of setup ids for which the param value equals the value passed

                    :param int param_id: Unique identifier for the param
                    :param int value: value of param
                    :return: A list of setup ids for which the constraint: param value equals 'value' exists
                    :rtype: List[int]
                    """
        try:

            self._execute("""SELECT DISTINCT setup.setup_id FROM reference_data_hist, data, setup WHERE 
            data.fk_session_id = setup.fk_session_id AND reference_data_hist.fk_data_id = data.data_id AND 
            reference_data_hist.fk_feature_param_id = %s AND reference_data_hist.bin = %s ORDER BY setup.setup_id""",
                          [param_id, str(value)])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_vs_in_range(self, param_id: int, min_value: int, max_value: int) -> Optional[list]:
        """ A list of setup ids for which the param value is between min and max values

                    :param int param_id: Unique identifier for the param
                    :param int min_value: value of param
                    :param int max_value: value of param
                    :return: A list of setup ids for which the constraint: param value between min and max exists
                    :rtype: List[int]
                    """
        try:

            self._execute("""SELECT DISTINCT setup.setup_id FROM reference_data_hist, data, setup WHERE 
            data.fk_session_id = setup.fk_session_id AND reference_data_hist.fk_data_id = data.data_id AND 
            fk_feature_param_id = %s AND bin >= %s AND bin <= %s ORDER BY setup.setup_id""",
                          [param_id, min_value, max_value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_id(self, data_id: int) -> Optional[List[int]]:
        """ Retrieve setup ids for which data_id exist

                    :param int data_id: Unique identifier for the data
                    :return: setup_ids for which data_id exist
                    :rtype: List[int]
                    """
        try:

            self._execute("""SELECT DISTINCT T.SETUP_ID FROM((SELECT setup.setup_id AS SETUP_ID FROM setup, data 
                                      WHERE setup.setup_id = data.fk_setup_id AND data.data_id = %s) 
                                      UNION ALL (SELECT setup.setup_id AS SETUP_ID FROM setup, session, data 
                                      WHERE setup.fk_session_id = session.session_id AND setup.fk_session_id = data.fk_session_id 
                                      AND data.data_id = %s))T
                                      ORDER BY T.SETUP_ID""", [data_id, data_id])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __data_path(self, data_id: int) -> Optional[str]:
        """ Retrieve __data_path for data_id

                    :param int data_id: Unique identifier for the data
                    :return: str __data_path for which data_id exist
                    :rtype: str
                    """
        try:

            self._execute("""SELECT fpath FROM data WHERE data_id = %s""", [data_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_vs_greater_than(self, vs: VS, value: int) -> Optional[List[int]]:
        """ List of setups for which the constraint: param value grater than 'min_value' exists

                      :param VS vs: vs name
                      :param int value: value of param
                      :return: list of setup ids
                      :rtype: List
                      """
        param_id = self.__feature_param_id(str(vs))
        if param_id:
            return self.__setup_vs_greater_than(param_id, value)
        else:
            return None

    def setup_vs_smaller_than(self, vs: VS, value: int) -> Optional[List[int]]:
        """ List of setups for which the constraint: param value smaller than 'max_value' exists

                      :param VS vs: vs name
                      :param int value: value of param
                      :return: list of setup ids
                      :rtype: List
                      """
        param_id = self.__feature_param_id(str(str(vs)))
        if param_id:
            return self.__setup_vs_smaller_than(param_id, value)
        else:
            return None

    def setup_vs_equals(self, vs: VS, value: int) -> Optional[List[int]]:
        """ List of setups for which param value equals the value passed

                      :param VS vs: vs name
                      :param int value: value of param
                      :return: list of setup ids
                      :rtype: List
                      """
        param_id = self.__feature_param_id(str(str(vs)))
        if param_id:
            return self.__setup_vs_equals(param_id, value)
        else:
            return None

    def setup_vs_in_range(self, vs: VS, minn: int, maxx: int) -> Optional[List[int]]:
        """ List of setups for which the param value is between min and max values

                    :param VS vs: vs name
                    :param int minn: value of param
                    :param int maxx: value of param
                    :return: list of setup ids
                    :rtype: List
                    """
        param_id = self.__feature_param_id(str(str(vs)))
        if param_id:
            return self.__setup_vs_in_range(param_id, minn, maxx)
        else:
            return None

    def _setup_uuid_by_id(self, setup_id: int) -> Optional[str]:
        """ Retrieve setup uuid for which setup_id exist

                       :param int setup_id: Unique identifier for the setup
                       :return: setup_uuid for which setup_id exist
                       :rtype: str
                       """
        try:
            self._execute("""SELECT setup_uuid FROM setup WHERE setup_id = %s""", [setup_id])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def _param_type_ids(self, param_type: str):
        """Retrieve parameter_tpye (such as 'benchmark', 'vs', 'gt') unique uuid and id"""
        try:
            self._execute(f"""
            SELECT feature_param_type_uuid, feature_param_type_id
            FROM feature_param_type 
            WHERE name = '{param_type}'""")
            res = self.cursor.fetchall()
            return res[0][0], res[0][1]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def _get_ids(self, table: str, entry: str, name_str='name', id_col_name=None):
        """Retrieve id and uuid"""
        if id_col_name is None:
            id_col_name = table
        uuid_str = f'{id_col_name}_uuid'
        id_str = f'{id_col_name}_id'
        return self._execute_and_fetch(f"""
            SELECT {uuid_str}, {id_str}
            FROM {table} 
            WHERE {name_str} = '{entry}'""")

    def _feature_param_ids(self, feature: str) -> tuple:
        """ Retrieve feature_param_id feature_param_uuid for which feature_param name exist

                :param str feature: feature name
                :return: (feature_param_id, feature_param_uuid) for which feature_param name exist
                :rtype: tuple
                """
        return self._execute_and_fetch(f"""
            SELECT feature_param_uuid, feature_param_id
            FROM feature_param
            WHERE  feature_param.name = '{feature}'""", 2)

    def _data_ids(self, setup: int, sensor: Sensor) -> (Optional[str], Optional[str]):
        res = self._execute_and_fetch(f"""
        SELECT data.data_uuid, data.data_id
        FROM data
         INNER JOIN sensor ON data.fk_sensor_id = sensor.sensor_id
         INNER JOIN setup ON setup.fk_session_id = data.fk_session_id
        WHERE  setup.setup_id = {setup}
          AND sensor.name = '{sensor}'
        """, 2)
        if isinstance(res[0], tuple):
            return res
        return [res]

    def setup_data_validity(self, setup: int, sensor: Sensor, vs: VS = None) -> Optional[str]:
        """ Retrieve data validity for which setup ID, sensor and optional vs exists

                :param int setup: setup ID
                :param Sensor sensor: sensor name
                :param VS vs: vs name
                :return: data validity name
                :rtype: str
                """
        if vs:
            return self.__setup_vs_validity(setup, sensor, vs)
        else:
            return self.__setup_sensor_validity(setup, sensor)

    def __setup_vs_validity(self, setup_id: int, sensor: Sensor, vs: VS) -> Optional[str]:
        """ Retrieve data validity for a given setup id for which sensor-vs exists

               :param int setup_id: setup ID
               :param Sensor sensor: sensor name
               :param VS vs: vs name
               :return: data validity name
               :rtype: str
               """
        try:
            self._execute("""SELECT data_validity.name FROM setup, data, data_validity, sensor, feature_param,
            join_data_validity_feature_param WHERE data.fk_session_uuid = setup.fk_session_uuid AND 
            join_data_validity_feature_param.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
            join_data_validity_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
            join_data_validity_feature_param.fk_data_uuid = data.data_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND setup.setup_id = %s AND sensor.name = %s AND 
            feature_param.name = %s""", [setup_id, sensor, vs])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            else:
                return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_sensor_validity(self, setup_id: int, sensor: Sensor) -> Optional[str]:
        """ Retrieve data validity for which setup_id and sensor exists

               :param int setup_id: setup ID
               :param Sensor sensor: sensor name
               :return: data validity name
               :rtype: str
               """
        if str(sensor) == 'nes':
            return self.__setup_nes_validity(setup_id, sensor)
        else:
            return self.__setup_ref_validity(setup_id, sensor)

    def __setup_nes_validity(self, setup_id: int, sensor: Sensor) -> Optional[str]:
        """ Retrieve data validity for which setup id and 'nes' sensor exists

               :param int setup_id: setup ID
               :param Sensor sensor: sensor name
               :return: data validity name
               :rtype: str
               """
        try:
            self._execute("""SELECT data_validity.name FROM setup, data, data_validity, sensor WHERE 
            data.fk_setup_id = setup.setup_id AND data.fk_sensor_uuid = sensor.sensor_uuid AND 
            data.fk_data_validity_uuid = data_validity.data_validity_uuid AND setup.setup_id = %s AND 
            sensor.name = %s""", [setup_id, sensor])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_ref_validity(self, setup_id: int, sensor: Sensor) -> Optional[str]:
        """ Retrieve data validity for which setup id and reference sensor exists

               :param int setup_id: setup ID
               :param Sensor sensor: sensor name
               :return: data validity name
               :rtype: str
               """
        try:
            self._execute("""SELECT data_validity.name FROM setup, data, data_validity, sensor WHERE 
            data.fk_session_uuid = setup.fk_session_uuid AND 
            data.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND setup.setup_id = %s AND sensor.name = %s""",
                          [setup_id, sensor])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_data_validity(self, value: Validation, sensor: Sensor = None, vs: VS = None) -> List[int]:
        """ Retrieve list of setup ids for which sensor, data validity and optional vs exists

                :param Sensor sensor: sensor name
                :param Validation value: validation name
                :param VS vs: vs name
                :return: list of setup ids
                :rtype: list
                """
        if vs:
            return self.__setups_vs_validity(sensor, vs, value)
        else:
            return self.__setups_sensor_validity(sensor, value)

    def __setups_vs_validity(self, sensor: Sensor, vs: VS, value: Validation) -> List[int]:
        """ Retrieve list of setup ids for which sensor-vs-validity exists

               :param Sensor sensor: sensor name
               :param Validation value: validity name
               :param VS vs: vs name
               :return: list of setup ids
               :rtype: List[int]
               """
        try:
            query = """SELECT setup.setup_id FROM setup, data, data_validity, sensor, feature_param,
                join_data_validity_feature_param WHERE data.fk_session_uuid = setup.fk_session_uuid AND 
                join_data_validity_feature_param.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
                join_data_validity_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
                join_data_validity_feature_param.fk_data_uuid = data.data_uuid AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND data_validity.name = '{0}' AND 
                feature_param.name = '{1}'""".format(value, vs)
            if sensor:
                query += """ AND sensor.name = '{0}'""".format(sensor)
            query += """ ORDER BY setup.setup_id"""
            self._execute(query)
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def __setups_sensor_validity(self, sensor: Sensor, value: Validation) -> List[int]:
        """ Retrieve list of setup ids for which sensor-validity exists

               :param Sensor sensor: sensor name
               :param Validation value: validity name
               :return: list of setup ids
               :rtype: List[int]
               """
        if sensor and str(sensor) == 'nes':
            return self.__setups_nes_validity(sensor, value)
        else:
            return self.__setups_ref_validity(sensor, value)

    def __setups_nes_validity(self, sensor: Sensor, value: Validation) -> List[int]:
        """ Retrieve list of setup ids for which 'nes' sensor validity exists

               :param Sensor sensor: sensor name
               :param Validation value: validity name
               :return: list of setup ids
               :rtype: List[int]
               """
        try:

            self._execute("""SELECT DISTINCT(data.fk_setup_id) FROM data, data_validity, sensor WHERE 
            data.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND sensor.name = %s AND data_validity.name = %s 
            ORDER BY data.fk_setup_id""", [sensor, value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def __setups_ref_validity(self, sensor: Sensor, value: Validation) -> List[int]:
        """ Retrieve list of setup ids for which reference sensor validity exists

               :param Sensor sensor: sensor name
               :param Validation value: validity name
               :return: list of setup ids
               :rtype: List[int]
               """
        try:
            if sensor:
                query = """SELECT DISTINCT(setup.setup_id) FROM setup, data, data_validity, sensor WHERE 
                data.fk_session_uuid = setup.fk_session_uuid AND 
                data.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND sensor.name = '{0}' AND data_validity.name = '{1}' 
                ORDER BY setup.setup_id""".format(sensor, value)
            else:
                query = """(SELECT DISTINCT(setup.setup_id) FROM setup, data, data_validity, sensor WHERE 
                data.fk_session_uuid = setup.fk_session_uuid AND 
                data.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND data_validity.name = '{0}' 
                ORDER BY setup.setup_id)
                UNION
                (SELECT DISTINCT(data.fk_setup_id) FROM data, data_validity, sensor WHERE 
                data.fk_data_validity_uuid = data_validity.data_validity_uuid AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND data_validity.name = '{0}'
                ORDER BY data.fk_setup_id)""".format(value)
            self._execute(query)
            res = self.cursor.fetchall()
            return sorted([row[0] for row in res])
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def _insert_param(self, ptype: str, name: str) -> int:
        param_type_uuid, param_type_id = self._param_type_ids(ptype)
        return self._execute_and_commit(f"""
                INSERT INTO `feature_param`(`fk_feature_param_type_uuid`, `fk_feature_param_type_id`, `name`)
                VALUES ('{param_type_uuid}', '{param_type_id}', '{name}')
                """)

    def set_benchmark(self, setup: int, benchmark: Benchmark, value: bool) -> bool:
        """ insert or remove setup from a benchmark. If the benchmark does not exists, it will be created
                :param int setup: Unique identifier for the setup
                :param Benchmark benchmark: Benchmark name
                :param bool value: add or remove
                :return: False/True
                :rtype: bool
                """
        setup_uuid = self._setup_uuid_by_id(setup)
        feature_uuid, feature_id = self._feature_param_ids(str(benchmark))
        if feature_uuid is None:
            self._insert_param('benchmark', str(benchmark))
            feature_uuid, feature_id = self._feature_param_ids(str(benchmark))
        if setup_uuid and feature_uuid:
            if not self.is_benchmark(setup, benchmark):
                return self._insert_join_setup_feature_param(setup_uuid, setup, feature_uuid, feature_id,
                                                             str(value).upper())
            else:
                return self.__update_join_setup_feature_param(setup_uuid, feature_uuid, str(value).upper())
        return False

    def is_benchmark(self, setup: int, benchmark: Benchmark) -> bool:
        """Check if benchmark type exists for a given setup

        :param int setup: Unique identifier for the setup
        :param Benchmark benchmark: Benchmark name
        :return: True/False
        :rtype: bool
        """
        try:

            self._execute("""SELECT join_setup_feature_param.* FROM join_setup_feature_param, 
            feature_param WHERE fk_setup_id = %s AND 
            join_setup_feature_param.fk_feature_param_id = feature_param.feature_param_id AND 
            feature_param.name = %s""", [setup, benchmark])
            count = self.cursor.rowcount
            return True if count > 0 else False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return False

    def _insert_join_setup_feature_param(self, fk_setup_uuid: str, fk_setup_id: int, fk_feature_param_uuid: str,
                                         fk_feature_param_id: int, value: str) -> bool:
        """ insert new benchmark to db

                                    :param str fk_setup_uuid: Universal Unique identifier for the setup
                                    :param int fk_setup_id: Unique identifier for the setup
                                    :param str fk_feature_param_uuid: Universal Unique identifier for the feature_param
                                    :param int fk_feature_param_id: Unique identifier for the feature_param
                                    :param str value: False/ True
                                    :return: False/True
                                    :rtype: bool
                                    """

        self._execute_and_commit(f"""
        INSERT INTO `join_setup_feature_param`(
        `fk_setup_uuid`, `fk_setup_id`,  `fk_feature_param_uuid`, `fk_feature_param_id`, `value`)
        VALUES ('{fk_setup_uuid}', '{fk_setup_id}', '{fk_feature_param_uuid}', '{fk_feature_param_id}', '{value}')
        """)

    def __update_join_setup_feature_param(self, fk_setup_uuid: str, fk_feature_param_uuid: str, value: str) -> bool:
        """ update benchmark-setup in db

                    :param str fk_setup_uuid: Universal Unique identifier for the setup
                    :param str fk_feature_param_uuid: Universal Unique identifier for the feature_param
                    :param str value: False/ True
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute(f"""
            UPDATE join_setup_feature_param
            SET value = '{value}'
            WHERE fk_setup_uuid = '{fk_setup_uuid}'
            AND fk_feature_param_uuid = '{fk_feature_param_uuid}'""")
            # Commit your changes in the database
            self.conn.commit()
            return affected_count > 0
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __sensor_ids(self, sensor: Sensor) -> tuple:
        """ Retrieve sensor UUID/ID for which sensor name exist

                    :param Sensor sensor:sensor name
                    :return: sensor UUID/ID
                    :rtype: tuple
                    """
        try:
            self._execute(
                """SELECT sensor_uuid, sensor_id FROM sensor WHERE name = %s""", [sensor])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def set_delay(self, setup: int, sensor: Sensor, value: float, search: Optional[str] = None) -> bool:
        """ set delay value by setup_id and sensor name

                    :param int setup: Unique identifier for the setup
                    :param Sensor sensor: sensor name
                    :param float value: delay value in sec
                    :param str search: Sensor file name to search
                    :return: False/True
                    :rtype: bool
                    """
        if str(sensor) not in NES_SENSORS and str(sensor) not in ADDITIONAL_SENSORS:
            setup_uuid = self._setup_uuid_by_id(setup)
            session_uuid = self.__session_uuid(setup)
            sensor_uuid = self.__sensor_ids(sensor)[0]
            list_data_ids = self.__data_ids(session_uuid, sensor_uuid, search)
            param_uuid, param_id = self._feature_param_ids('nes_ref_delay')
            if setup_uuid and session_uuid and sensor_uuid and list_data_ids and param_uuid:
                for data in list_data_ids:
                    if self.__delay_exists(setup_uuid, data[0]):
                        return self._update_delay(setup, sensor, value)
                    else:
                        return self.__insert_delay(setup_uuid, setup, data[0], data[1], param_uuid, param_id, value)

        return False

    def _insert(self, query):
        try:
            self._execute(query)
            self.conn.commit()  # Commit your changes in the database
            return True
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            self.conn.rollback()
            return False

    def insert_subject(self, name, subject_type='Human', company='Neteera', birth_year=None, height_cm=None,
                       weight_kg=None, gender=None, note=None):
        type_uuid, type_id = self._get_ids('subject_type', subject_type)
        company_uuid, company_id = self._get_ids('company', company)
        query_insert_start = f""" # noqa
        INSERT INTO subject # noqa
        (fk_subject_type_uuid, fk_subject_type_id, fk_company_uuid, fk_company_id, name"""  # noqa
        query_insert_end = f""")  VALUES ('{type_uuid}', '{type_id}', '{company_uuid}', '{company_id}', '{name}' """
        for col_name in ['birth_year', 'height_cm', 'weight_kg', 'gender', 'note']:
            col_value = locals()[col_name]
            if col_value is not None:
                query_insert_start += f', {col_name}'
                query_insert_end += f", '{col_value}'"
        return self._insert(query_insert_start + query_insert_end + ')')

    def insert_sn(self, sn: int, radar_type: str, company: str, note=None):
        component_uuid, component_id = db._get_ids('component', radar_type)
        company_uuid, company_id = db._get_ids('company', company)
        self._insert(f"""
        INSERT INTO component_inventory
        (fk_component_uuid, fk_component_id, fk_company_uuid, fk_company_id,  serial_num, note)
        VALUES ('{component_uuid}', '{component_id}', '{company_uuid}', '{company_id}', '{sn}', '{note}')""")

    def insert_ref_path(self, setup: int, sensor: Sensor, path: str, data=None, validity=Validation.confirmed) -> bool:
        print(f'saving ref file to db setup {setup} \n {path}')

        os.system('chmod 775 -R ' + path)
        path_for_db = get_db_path(path)
        setup_uuid, _ = self._get_ids('setup', setup, 'setup_id')
        session_uuid, session_id = self._get_ids('setup', setup, 'setup_id', id_col_name='fk_session')
        sensor_uuid, sensor_id = self._get_ids('sensor', str(sensor))
        validity_uuid, validity_id = self.__data_validity_ids(str(validity))
        self._insert(f"""INSERT INTO `data`(
        `fk_session_uuid`, `fk_session_id`, `fk_setup_uuid`, `fk_setup_id`, `fk_sensor_uuid`,
        `fk_sensor_id`, `fpath`, `fk_data_validity_uuid`, `fk_data_validity_id`) 
        VALUES ('{session_uuid}', {session_id}, '{setup_uuid}', {setup}, '{sensor_uuid}', {sensor_id},
                '{path_for_db}', '{validity_uuid}', {validity_id})""")

    def insert_npy_ref(self, setup: int, sensor: Sensor, vs: VS, path: str, data) -> bool:
        print(f'saving ref file to db setup {setup} vs {vs}\n {path}')

        os.system('chmod 775 -R ' + path)
        count = 0
        try:
            data.to_pickle(path)
        except AttributeError:
            np.save(path, data)
        path_for_db = get_db_path(path)
        fk_feature_param_uuid, fk_feature_param_id = self._feature_param_ids(vs)
        data_ids = self._data_ids(setup, sensor)
        str_hash = str(path_for_db)
        sha = sha256(str_hash.encode('utf-8')).hexdigest()
        for fk_data_uuid, fk_data_id in data_ids:
            count += self._insert(f"""
            REPLACE INTO data_npy
            (fk_data_uuid, fk_data_id, fk_feature_param_uuid,fk_feature_param_id, fpath, sha)
            VALUES (
            '{fk_data_uuid}', {fk_data_id}, '{fk_feature_param_uuid}', {fk_feature_param_id}, '{path_for_db}', '{sha}'
            )""")
        return count

    def __insert_delay(self, setup_uuid: str, setup_id: int, data_uuid: str, data_id: int, param_uuid: str,
                       param_id: int, value: float) -> bool:
        """ insert nes ref delay to db
                :param str setup_uuid: Universal Unique identifier for the setup
                :param int setup_id: Unique identifier for the setup
                :param str data_uuid: Universal Unique identifier for the data
                :param int data_id: Unique identifier for the data
                :param str param_uuid: Universal Unique identifier for the param
                :param int param_id: Unique identifier for the param
                :param float value: delay value
                :return: False/True
                :rtype: bool
                """
        try:
            self._execute("""INSERT INTO `join_setup_data_param`(`fk_setup_uuid`, `fk_setup_id`, `fk_data_uuid`, 
            `fk_data_id`, `fk_feature_param_uuid`, `fk_feature_param_id`, `value`) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                          (setup_uuid, setup_id, data_uuid, data_id, param_uuid, param_id, value))
            # Commit your changes in the database
            self.conn.commit()
            return True
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def _update_delay(self, setup, sensor, shift) -> bool:
        """ update delay field in 'join_setup_data_param' table of db

               :return: False/True
               :rtype: bool
               """
        try:
            affected_count = self._execute(f"""
            UPDATE join_setup_data_param
                 INNER JOIN feature_param
                    ON join_setup_data_param.fk_feature_param_uuid = feature_param.feature_param_uuid
                 INNER JOIN setup ON join_setup_data_param.fk_setup_id = setup.setup_id
                 INNER JOIN data ON join_setup_data_param.fk_data_uuid = data.data_uuid
                 INNER JOIN sensor ON data.fk_sensor_uuid = sensor.sensor_uuid
            SET value = {shift}
            WHERE feature_param.name = 'nes_ref_delay'
              AND setup.setup_id = '{setup}'
              AND sensor.name = '{sensor}'""")
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __data_validity_ids(self, name: Validation) -> tuple:
        """ return id and uuid for a given validation name

                            :param Validation name: 'Valid'/ 'Invalid'/ 'Confirmed'
                            :return: (data_validity_id, data_validity_uuid)
                            :rtype: tuple
                            """
        return self._execute_and_fetch(f"""
        SELECT data_validity_uuid, data_validity_id
        FROM data_validity
        WHERE name = '{name}'""")

    def __data_ids(self, session_uuid: str, sensor_uuid: str, search: Optional[str] = None) -> List[tuple]:
        """ return data UUID and ID for a given session ID and sensor name

                :param str session_uuid: session UUID
                :param str sensor_uuid: sensor UUID
                :param str search: Sensor file name to search
                :return: data UUID
                :rtype: str
                """
        try:
            query = """SELECT data_uuid, data_id FROM data WHERE fk_session_uuid = '{}' AND 
            fk_sensor_uuid = '{}'""".format(session_uuid, sensor_uuid)
            if search:
                query += """ AND fpath LIKE '{}'""".format('%' + search + '%')
            self._execute(query)
            res = self.cursor.fetchall()
            return [(row[0], row[1]) for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def set_data_validity(self, setup: int, sensor: Sensor, value: Validation, vs: VS = None) -> bool:
        """ set data validity by setup_id and sensor name and optional vs

        :param int setup: Unique identifier for the setup
        :param Sensor sensor: sensor
        :param Validation value: 'Valid'/ 'Invalid'/ 'Confirmed'
        :param VS vs: VS name
        :return: False/True
        :rtype: bool
        """
        if value == Validation.invalid:
            for x in self.setup_multi(setup):
                for bench in Benchmark:
                    self.set_benchmark(x, bench, False)
                note = f'{self.setup_note(setup)}; invalid {sensor}'
                if vs is not None:
                    note += f' vs {vs}'
            self.set_note(setup, note)
        if vs:
            return self.__set_vs_validity(setup, sensor, vs, value)
        else:
            return self.__set_sensor_validity(setup, sensor, value)

    def __set_sensor_validity(self, setup: int, sensor: Sensor, value: Validation) -> bool:
        """ set data validity by setup_id and sensor name

                    :param int setup: Unique identifier for the setup
                    :param Sensor sensor: sensor name
                    :param Validation value: 'Valid'/ 'Invalid'/ 'Confirmed'
                    :return: False/True
                    :rtype: bool
                    """
        data_validity_uuid, data_validity_id = self.__data_validity_ids(value)
        sensor_uuid = self.__sensor_ids(sensor)[0]
        if str(sensor) != 'nes':
            session_id = self.__session_id(setup)
            if None not in [sensor_uuid, session_id, data_validity_uuid, data_validity_id]:
                return self.__update_data_validity(data_validity_uuid, data_validity_id, session_id, sensor_uuid)
        else:
            if None not in [sensor_uuid, data_validity_uuid, data_validity_id]:
                return self.__update_nes_validity(data_validity_uuid, data_validity_id, setup, sensor_uuid)
        return False

    def __set_vs_validity(self, setup: int, sensor: Sensor, vs: VS, value: Validation) -> bool:
        """ insert new vital sign validity for a given setup_id and sensor name

                :param int setup: Unique identifier for the setup
                :param Sensor sensor: Sensor name
                :param VS vs: VS name
                :param Validation value: 'Valid'/ 'Invalid'/ 'Confirmed'
                :return: False/True
                :rtype: bool
                """
        if str(sensor) not in NES_SENSORS and str(sensor) not in ADDITIONAL_SENSORS:
            file_data = None
            if sensor == Sensor.epm_10m:
                if vs == VS.hri:
                    file_data = 'ECG'
                else:
                    file_data = 'ParameterData'
            sensor_uuid = self.__sensor_ids(sensor)[0]
            session_uuid = self.__session_uuid(setup)
            data_validity_uuid, data_validity_id = self.__data_validity_ids(value)
            if session_uuid and sensor_uuid and data_validity_uuid:
                list_data_ids = self.__data_ids(session_uuid, sensor_uuid, file_data)
                vs_uuid, vs_id = self._feature_param_ids(str(vs))
                if list_data_ids and vs_uuid:
                    for data in list_data_ids:
                        if self.vs_validity_exists(data[0], vs_uuid):
                            return self.__update_join_data_validity_feature_param(data[0], vs_uuid, data_validity_uuid,
                                                                                  data_validity_id)
                        else:
                            return self.__insert_join_data_validity_feature_param(data[0], data[1], vs_uuid, vs_id,
                                                                                  data_validity_uuid, data_validity_id)
        return False

    def vs_validity_exists(self, data_uuid: str, vs_uuid: str) -> bool:
        """Check if vs validation exists for a given data UUID

        :param str data_uuid: data UUID
        :param str vs_uuid: vs UUID
        :return: True/False
        :rtype: bool
        """
        try:

            self._execute("""SELECT join_data_validity_feature_param.* FROM join_data_validity_feature_param 
            WHERE fk_data_uuid = %s AND fk_feature_param_uuid = %s""", [data_uuid, vs_uuid])
            count = self.cursor.rowcount
            return True if count > 0 else False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return False

    def __delay_exists(self, setup_uuid: str, data_uuid: str) -> Optional[bool]:
        """Check if delay value exists for a given setup and data UUID

        :param str setup_uuid: setup UUID
        :param str data_uuid: data UUID
        :return: True/False
        :rtype: bool
        """
        try:

            self._execute("""SELECT join_setup_data_param.* FROM join_setup_data_param, feature_param WHERE 
            join_setup_data_param.fk_setup_uuid = %s AND join_setup_data_param.fk_data_uuid = %s AND 
            join_setup_data_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
            feature_param.name = 'nes_ref_delay'""", [setup_uuid, data_uuid])
            count = self.cursor.rowcount
            return True if count > 0 else False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __insert_join_data_validity_feature_param(self, data_uuid: str, data_id: int, param_uuid: str,
                                                  param_id: int, validity_uuid: str, validity_id: int) -> bool:
        """ insert new vs validity to db

                :param str data_uuid: Universal Unique identifier for the data
                :param int data_id: Unique identifier for the data
                :param str param_uuid: Universal Unique identifier for the feature_param
                :param int param_id: Unique identifier for the feature_param
                :param str validity_uuid: Universal Unique identifier for the data validity
                :param int validity_id: Unique identifier for the data validity
                :return: False/True
                :rtype: bool
                """
        try:

            self._execute("""INSERT INTO `join_data_validity_feature_param`(`fk_data_uuid`, `fk_data_id`, 
                        `fk_feature_param_uuid`, `fk_feature_param_id`, `fk_data_validity_uuid`,
                         `fk_data_validity_id`) VALUES (%s, %s, %s, %s, %s, %s)""",
                          (data_uuid, data_id, param_uuid, param_id, validity_uuid, validity_id))
            # Commit your changes in the database
            self.conn.commit()
            return True
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __update_join_data_validity_feature_param(self, data_uuid: str, param_uuid: str, validity_uuid: str,
                                                  validity_id: int) -> bool:
        """ update vs validity in db

                :param str data_uuid: Universal Unique identifier for the data
                :param str param_uuid: Universal Unique identifier for the feature_param
                :param str validity_uuid: Universal Unique identifier for the data validity
                :param int validity_id: Unique identifier for the data validity
                :return: False/True
                :rtype: bool
                """
        try:
            affected_count = self._execute("""UPDATE join_data_validity_feature_param SET fk_data_validity_uuid = %s, 
            fk_data_validity_id = %s WHERE fk_data_uuid = %s AND fk_feature_param_uuid = %s""",
                                           [validity_uuid, validity_id, data_uuid, param_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __update_data_validity(self, fk_data_validity_uuid: str, fk_data_validity_id: int, session_id: int,
                               sensor_uuid: str) -> bool:
        """ set data validity by session id and sensor uuid

                     :param str fk_data_validity_uuid: Unique identifier for the validation param
                     :param int fk_data_validity_id: UUID for the validation param
                     :param int session_id: Unique identifier for the session
                     :param str sensor_uuid: UUID for the sensor param
                     :return: False/True
                     :rtype: bool
                     """
        try:
            affected_count = self._execute("""UPDATE data SET fk_data_validity_uuid = %s, fk_data_validity_id = %s 
            WHERE fk_session_id = %s AND fk_sensor_uuid = %s""", [fk_data_validity_uuid, fk_data_validity_id,
                                                                  session_id, sensor_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __update_nes_validity(self, fk_data_validity_uuid: str, fk_data_validity_id: int, setup_id: int,
                              sensor_uuid: str) -> bool:
        """ set data validity by setup id and sensor uuid

                     :param str fk_data_validity_uuid: Unique identifier for the validation param
                     :param int fk_data_validity_id: UUID for the validation param
                     :param int setup_id: Unique identifier for the setup
                     :param str sensor_uuid: UUID for the sensor param
                     :return: False/True
                     :rtype: bool
                     """
        try:
            affected_count = self._execute("""UPDATE data SET fk_data_validity_uuid = %s, fk_data_validity_id = %s 
            WHERE fk_setup_id = %s AND fk_sensor_uuid = %s""", [fk_data_validity_uuid, fk_data_validity_id, setup_id,
                                                                sensor_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def setup_vs_avg(self, setup: int, sensor: Sensor, vs: VS) -> Optional[float]:
        """ return the average value of a vs for a given setup ID, sensor name and vs name

                :param: int setup: Unique identifier for the setup
                :param: Sensor sensor: sensor name
                :param: VS vs: vs name
                :return: vs avg
                :rtype: float
                """
        try:

            self._execute("""SELECT join_data_feature_param.value FROM session, setup, sensor, data, feature_param, 
            join_data_feature_param WHERE setup.fk_session_uuid = session.session_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND data.fk_session_uuid = session.session_uuid AND 
            join_data_feature_param.fk_data_uuid = data.data_uuid AND 
            join_data_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND setup.setup_id = %s AND 
            sensor.name = %s AND feature_param.name = %s""", [setup, str(sensor), str(vs) + '_avg'])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def set_note(self, setup: int, note: str) -> bool:
        """ set session note by setup id

            :param int setup: Unique identifier for the setup
            :param str note: session note to be modified
            :return: False/True
            :rtype: bool
            """
        try:
            affected_count = self._execute("""UPDATE session INNER JOIN setup ON 
                                  session.session_uuid = setup.fk_session_uuid AND setup.setup_id = %s 
                                  SET session.note = %s""", [setup, note])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_company(self, setup: int, prj: Project) -> bool:
        """ set company for a given setup id

            :param int setup: Unique identifier for the setup
            :param Project prj: Company name
            :return: False/True
            :rtype: bool
            """
        company_uuid, company_id = self.__company_ids(prj)
        if company_uuid:
            return self.__update_company(setup, company_uuid, company_id)
        return False

    def __company_ids(self, prj: Project) -> tuple:
        """ get company ids

                :param Posture prj: company name
                :return: Company ids
                :rtype: tuple
                """
        try:
            affected_count = self._execute("""SELECT company_uuid, company_id FROM company WHERE name = %s""", [prj])
            res = self.cursor.fetchall()
            if affected_count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __update_company(self, setup: int, company_uuid: str, company_id: int) -> Optional[bool]:
        """ update session company

                :param int setup: Unique identifier for the setup
                :param str company_uuid: Company UUID
                :param int company_id: Company ID
                return: False/True
                :rtype: bool
                """
        try:
            affected_count = self._execute("""UPDATE session INNER JOIN setup ON 
            session.session_uuid = setup.fk_session_uuid AND setup.setup_id = %s SET 
            session.fk_company_uuid = %s, session.fk_company_id = %s""", [setup, company_uuid, company_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_fs(self, setup: int, fs: float) -> bool:
        """ set setup fs by setup id

            :param int setup: Unique identifier for the setup
            :param float fs: setup frame rate
            :return: False/True
            :rtype: bool
            """
        return self._execute_and_commit(f"""UPDATE join_nes_config_param 
            INNER JOIN nes_param ON join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid 
            INNER JOIN nes_config ON join_nes_config_param.fk_nes_config_uuid = nes_config.nes_config_uuid
            INNER JOIN setup ON nes_config.nes_config_uuid = setup.fk_nes_config_uuid 
            AND nes_param.name = 'infoFW_framerate' AND setup.setup_id = {setup}
            SET join_nes_config_param.value = {fs}""")

    def setup_by_epoch(self, epoch: str) -> Optional[List[int]]:
        """ return setup id for a given epoch ts

                :param str epoch: epoch time stamp
                :return: setup id
                :rtype: int
                """
        try:

            self._execute("""SELECT DISTINCT(setup.setup_id) FROM setup, data WHERE data.fk_setup_id = setup.setup_id
            AND `fpath` LIKE %s ORDER BY setup.setup_id""", ["%" + epoch + "%"])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_ts(self, setup: int, sensor: Sensor, search: Optional[str] = None) -> \
            Optional[Dict[str, Optional[datetime.datetime]]]:
        """ return starting/ending ts for a given setup id and sensor name

                :param int setup: setup id
                :param Sensor sensor: Sensor name
                :param str search: Sensor file name to search
                :return: starting/ending timestamp
                :rtype: dict
                """
        sch_str = ref_str = ''
        if search:
            sch_str = """ AND data.fpath LIKE '%{sch}%'""".format(sch=search)
        else:
            if sensor == Sensor.nes and self.mysql_db == 'neteera_db':
                ref_str = """ AND (LENGTH(data.fpath)-LENGTH(REPLACE(data.fpath,'.','')))/LENGTH('.') = 1 AND 
                ((substring_index(data.fpath, '.', -1) = 'ttlog') OR 
                (substring_index(data.fpath, '.', -1) = 'tlog') OR 
                (substring_index(data.fpath, '.', -1) = 'blog'))"""
            elif sensor == Sensor.epm_10m:
                ref_str = """ AND data.fpath LIKE '%ParameterData%'"""
            elif sensor == Sensor.natus:
                ref_str = """ AND data.fpath LIKE '%.edf%'"""
        query = f"""
        SELECT data.start_time, data.end_time
        FROM data, sensor
         WHERE data.fk_setup_id = {setup}
         AND data.fk_sensor_uuid = sensor.sensor_uuid
         AND sensor.name = '{sensor}' {sch_str} {ref_str}
         AND data.start_time IS NOT NULL
        UNION 
        SELECT data.start_time, data.end_time FROM data, sensor, setup WHERE setup.setup_id = {setup} AND 
        data.fk_session_uuid = setup.fk_session_uuid AND data.fk_sensor_uuid = sensor.sensor_uuid AND 
        sensor.name = '{sensor}' {sch_str} {ref_str}"""
        start, end = self._execute_and_fetch(query, 2)
        if start is not None:
            if self.mysql_db == 'neteera_cloud_mirror':
                time_zone = 'utc'
            else:
                time_zone = self.get_setup_time_zone(setup)
                if time_zone == 'None' or str(sensor) == 'nes':
                    time_zone = 'Asia/Jerusalem'
            pytz_zone = pytz.timezone(time_zone).localize(start).tzinfo
            return {'start': start.replace(tzinfo=pytz_zone), 'end': end.replace(tzinfo=pytz_zone)}

    def set_setup_time_zone(self, setup: int, time_zone: str):
        self._execute_and_commit(f"""
        UPDATE session
        INNER JOIN setup ON session.session_id = setup.fk_session_id
        SET session.timezone = '{time_zone}'
        WHERE setup.setup_id  = {setup}
        """)

    def get_setup_time_zone(self, setup: int):
        return self._execute_and_fetch(f"""
            SELECT session.timezone
            FROM session
            INNER JOIN setup ON session.session_id = setup.fk_session_id
            WHERE setup.setup_id  = {setup}
            """)

    def setup_by_project(self, prj: Project) -> List[int]:
        """ Retrieve setup ids for which project name exist

            :param Project prj: project name
            :return: List of setup ids
            :rtype: List[int]
        """
        return self._execute_and_fetch(f"""
        SELECT setup.setup_id
        FROM session, setup, company
        WHERE setup.fk_session_uuid = session.session_uuid
        AND session.fk_company_uuid = company.company_uuid
        AND  company.name = '{prj}' ORDER BY setup.setup_id""")

    def setup_by_company(self, comp: Project) -> List[int]:
        return self.setup_by_project(comp)

    def setup_rest_back(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) cw (1) rest sessions (labeled rest by FAEs) with (2) no motion, (3) no engine on,
         (4) no speaking and (5) not driving/driving_idle with (6) back mount (7) has reference excluding
         'elbit_ecg_sample' (8) nes validity is 'valid' or 'confirmed' (9) fs>0
                :return: List of setup ids
                :rtype: List[int]
        """
        cw = self.sr_cw_setups()
        rest_sess = self.setup_by_state(state=State.is_rest, value=True)
        motion_sess = self.setup_by_state(state=State.is_motion, value=True)
        engine_on_sess = self.setup_by_state(state=State.is_engine_on, value=True)
        speaking_sess = self.setup_by_state(state=State.is_speaking, value=True)
        driving_sess = self.setup_by_state(state=State.is_driving, value=True)
        driving_idle_sess = self.setup_by_state(state=State.is_driving_idle, value=True)
        back_mount_sess = self.setup_by_target(target=Target.back)
        if None not in [cw, rest_sess, motion_sess, engine_on_sess, speaking_sess, driving_sess, driving_idle_sess,
                        back_mount_sess]:
            setups = list(set(cw) & set(rest_sess) & set(back_mount_sess))
            setups_exclude = list(set(motion_sess) | set(engine_on_sess) | set(speaking_sess) | set(driving_sess) |
                                  set(driving_idle_sess))
            setups = [se for se in setups if se not in setups_exclude]
            valid_setups = []
            for se in setups:
                nes_validity = self.setup_data_validity(setup=se, sensor=Sensor.nes)
                fs = self.setup_fs(se)
                if None not in [nes_validity, fs]:
                    if nes_validity not in ['Invalid'] and fs > 0:
                        valid_setups.append(se)
                else:
                    return None
            setups_with_ref = []
            for s in valid_setups:
                ref = self.setup_ext_proc_ref(s)
                if ref is not None:
                    if not (len(ref) == 1 and Sensor.elbit_ecg_sample.name.upper() in ref):
                        setups_with_ref.append(s)
                else:
                    return None
            return sorted(setups_with_ref)
        return None

    def setup_motion_back(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) cw (1) motion sessions (labeled motion by FAEs) with, (2) no engine on,
         (3) no speaking and (4) not driving/driving_idle with (5) back mount (6) has reference excluding
         'elbit_ecg_sample' (7) nes validity is 'valid' or 'confirmed' (8) fs>0
                :return: List of setup ids
                :rtype: List[int]
        """
        cw = self.sr_cw_setups()
        motion_sess = self.setup_by_state(state=State.is_motion, value=True)
        engine_on_sess = self.setup_by_state(state=State.is_engine_on, value=True)
        speaking_sess = self.setup_by_state(state=State.is_speaking, value=True)
        driving_sess = self.setup_by_state(state=State.is_driving, value=True)
        driving_idle_sess = self.setup_by_state(state=State.is_driving_idle, value=True)
        back_mount_sess = self.setup_by_target(target=Target.back)
        if None not in [cw, motion_sess, engine_on_sess, speaking_sess, driving_sess, driving_idle_sess,
                        back_mount_sess]:
            setups = list(set(cw) & set(motion_sess) & set(back_mount_sess))
            setups_exclude = list(set(engine_on_sess) | set(speaking_sess) | set(driving_sess) | set(driving_idle_sess))
            setups = [se for se in setups if se not in setups_exclude]
            valid_setups = []
            for se in setups:
                nes_validity = self.setup_data_validity(setup=se, sensor=Sensor.nes)
                fs = self.setup_fs(se)
                if None not in [nes_validity, fs]:
                    if nes_validity not in ['Invalid'] and fs > 0:
                        valid_setups.append(se)
                else:
                    return None
            setups_with_ref = []
            for s in valid_setups:
                ref = self.setup_ext_proc_ref(s)
                if ref is not None:
                    if not (len(ref) == 1 and Sensor.elbit_ecg_sample.name.upper() in ref):
                        setups_with_ref.append(s)
                else:
                    return None
            return sorted(setups_with_ref)
        return None

    def setup_driving_back(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) cw, (1) all sorts of driving, (2) no speaking and with (3) back mount (4) has
         reference excluding 'elbit_ecg_sample' (5) nes validity is 'valid' or 'confirmed' (6) fs>0
                :return: List of setup ids
                :rtype: List[int]
        """
        cw = self.sr_cw_setups()
        driving_sess = self.setup_by_state(state=State.is_driving, value=True)
        driving_idle_sess = self.setup_by_state(state=State.is_driving_idle, value=True)
        all_driving_sess = set(driving_sess) | set(driving_idle_sess) if None not in [driving_sess, driving_idle_sess] \
            else None
        speaking_sess = self.setup_by_state(state=State.is_speaking, value=True)
        back_mount_sess = self.setup_by_target(target=Target.back)
        if None not in [cw, all_driving_sess, speaking_sess, back_mount_sess]:
            setups = list(set(cw) & all_driving_sess & set(back_mount_sess))
            setups = [se for se in setups if se not in speaking_sess]
            valid_setups = []
            for se in setups:
                nes_validity = self.setup_data_validity(setup=se, sensor=Sensor.nes)
                fs = self.setup_fs(se)
                if None not in [nes_validity, fs]:
                    if nes_validity not in ['Invalid'] and fs > 0:
                        valid_setups.append(se)
                else:
                    return None
            setups_with_ref = []
            for s in valid_setups:
                ref = self.setup_ext_proc_ref(s)
                if ref is not None:
                    if not (len(ref) == 1 and Sensor.elbit_ecg_sample.name.upper() in ref):
                        setups_with_ref.append(s)
                else:
                    return None
            return sorted(setups_with_ref)
        return None

    def setup_for_test_4(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) cw  (1) cherry (2) rest sessions (labeled rest by FAEs) with (3) no motion,
         (4) no engine on, (5) no speaking and (6) not driving/driving_idle with (7) back mount (8) has reference
         excluding 'elbit_ecg_sample' (9) nes validity is 'valid' or 'confirmed' (10) fs>0 (11) agc=0 (12) fw >='0.4.6.2'
                :return: List of setup ids
                :rtype: List[int]
        """
        cw = self.sr_cw_setups()
        cherry = self.setup_by_model(model='cherry')
        rest_sess = self.setup_by_state(state=State.is_rest, value=True)
        motion_sess = self.setup_by_state(state=State.is_motion, value=False)
        engine_on_sess = self.setup_by_state(state=State.is_engine_on, value=False)
        speaking_sess = self.setup_by_state(state=State.is_speaking, value=False)
        driving_sess = self.setup_by_state(state=State.is_driving, value=False)
        driving_idle_sess = self.setup_by_state(state=State.is_driving_idle, value=False)
        back_mount_sess = self.setup_by_target(target=Target.back)
        agc = self.agc_setups()
        hr_sess = self.setup_by_vs(VS.hr)
        ie_invalid_setups = {2423, 2422, 2420}  # inhale exhale unnatural
        if None not in [cw, cherry, rest_sess, motion_sess, engine_on_sess, speaking_sess, driving_sess,
                        driving_idle_sess, back_mount_sess, agc, hr_sess]:
            setups = list(set(cw) & set(cherry) & set(rest_sess) & set(motion_sess) & set(engine_on_sess) &
                          set(speaking_sess) & set(driving_sess) & set(driving_idle_sess) & set(back_mount_sess) &
                          set(hr_sess))
            valid_setups = []
            for se in setups:
                nes_validity = self.setup_data_validity(setup=se, sensor=Sensor.nes)
                fs = self.setup_fs(se)
                fw = self.setup_fw(se)
                if None not in [nes_validity, fs, fw]:
                    if nes_validity not in ['Invalid'] and fs > 0 and (version.parse(fw) >= version.parse("0.4.6.2")):
                        valid_setups.append(se)
                else:
                    return None
            setups_exclude = list(set(agc) | ie_invalid_setups)
            valid_setups = [se for se in valid_setups if se not in setups_exclude]
            setups_with_ref = []
            for s in valid_setups:
                ref = self.setup_ext_proc_ref(s)
                if ref is not None:
                    ref = [r.lower() for r in ref if r.lower() not in GT_SENSORS]
                    if not (len(ref) == 1 and Sensor.elbit_ecg_sample.name.upper() in ref):
                        for r in ref:
                            ref_validity = self.setup_data_validity(setup=s, sensor=Sensor[r], vs=VS.hr)
                            if ref_validity is not None:
                                if ref_validity not in ['Invalid'] and r != Sensor.elbit_ecg_sample.name:
                                    if s not in setups_with_ref:
                                        setups_with_ref.append(s)
                else:
                    return None
            return sorted(setups_with_ref)
        return None

    def setup_fm_cw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw  (1) nes validity is 'valid' or 'confirmed'
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        fmcw = self._remove_raw_cpx(fmcw)
        if fmcw is not None:
            valid_setups = []
            for s in fmcw:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                if nes_validity:
                    if nes_validity not in ['Invalid']:
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_raw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) raw data
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        raw = self.setup_by_vs(vs=VS.raw)
        if None not in [fmcw, raw]:
            fmcw_rwa = list(set(fmcw) & set(raw))
            fmcw_rwa = self._remove_raw_cpx(fmcw_rwa)
            valid_setups = []
            for s in fmcw_rwa:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                if nes_validity:
                    if nes_validity not in ['Invalid']:
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_cpx(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        if None not in [fmcw, cpx]:
            fmcw_cpx = list(set(fmcw) & set(cpx))
            fmcw_cpx = self._remove_raw_cpx(fmcw_cpx)
            valid_setups = []
            for s in fmcw_cpx:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                if nes_validity:
                    if nes_validity not in ['Invalid']:
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_raw_front(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) raw data (3) front
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        raw = self.setup_by_vs(vs=VS.raw)
        if None not in [fmcw, raw]:
            fmcw_rwa = list(set(fmcw) & set(raw))
            fmcw_rwa = self._remove_raw_cpx(fmcw_rwa)
            valid_setups = []
            for s in fmcw_rwa:
                target = self.setup_target(setup=s)
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                if nes_validity and target:
                    if nes_validity not in ['Invalid'] and target == 'front':
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_raw_back(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) raw data (3) back
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        raw = self.setup_by_vs(vs=VS.raw)
        if None not in [fmcw, raw]:
            fmcw_rwa = list(set(fmcw) & set(raw))
            fmcw_rwa = self._remove_raw_cpx(fmcw_rwa)
            valid_setups = []
            for s in fmcw_rwa:
                target = self.setup_target(setup=s)
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                if nes_validity and target:
                    if nes_validity not in ['Invalid'] and target == 'back':
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_cpx_front(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) front
        (4) fw >='0.4.7.5'
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        if None not in [fmcw, cpx]:
            fmcw_cpx = list(set(fmcw) & set(cpx))
            fmcw_cpx = self._remove_raw_cpx(fmcw_cpx)
            valid_setups = []
            for s in fmcw_cpx:
                target = self.setup_target(setup=s)
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and target and fw:
                    if nes_validity not in ['Invalid'] and target == 'front' and \
                            (version.parse(fw) >= version.parse("0.4.7.5")):
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw_cpx_back(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) back
        (4) fw >='0.4.7.5'
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        if None not in [fmcw, cpx]:
            fmcw_cpx = list(set(fmcw) & set(cpx))
            fmcw_cpx = self._remove_raw_cpx(fmcw_cpx)
            valid_setups = []
            for s in fmcw_cpx:
                target = self.setup_target(setup=s)
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and target and fw:
                    if nes_validity not in ['Invalid'] and target == 'back' and \
                            (version.parse(fw) >= version.parse("0.4.7.5")):
                        valid_setups.append(s)
                else:
                    return None
            return valid_setups
        return None

    def setup_cpp_fmcw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) front or back
        (4) fw>='0.4.7.5' (5) EC=0
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        occ_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        if None not in [occ_sess, ec_sess]:
            inter_sess = set(occ_sess) & set(ec_sess)
            valid_sess = set(occ_sess) - inter_sess
        else:
            return None
        if None not in [fmcw, cpx]:
            fmcw_cpx = list(set(fmcw) & set(cpx) & valid_sess)
            fmcw_cpx = self._remove_raw_cpx(fmcw_cpx)
            valid_setups = []
            for s in fmcw_cpx:
                target = self.setup_target(setup=s)
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and target and fw:
                    if nes_validity not in ['Invalid'] and (target == 'front' or target == 'back') and \
                            (version.parse(fw) >= version.parse("0.4.7.5")):
                        valid_setups.append(s)
                else:
                    return None
            return sorted(valid_setups)
        return None

    def _remove_raw_cpx(self, setups: list) -> list:
        raw = self.setup_by_vs(vs=VS.raw)
        cpx = self.setup_by_vs(vs=VS.cpx)
        return set(setups) - (set(raw) & set(cpx))

    def setup_status_fmcw_cpx(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) fw>='0.4.7.5'
        (4) EC=0&1 or ZRR=0&1
           :return: List of setup ids
           :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        raw = self.setup_by_vs(vs=VS.raw)
        cpx = self.setup_by_vs(vs=VS.cpx)
        full_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess = self.setup_vs_equals(VS.zrr, 1)
        resp_sess = self.setup_vs_equals(VS.zrr, 0)
        if None not in [full_sess, ec_sess, resp_sess, zrr_sess]:
            occ_sess = set(full_sess) & set(ec_sess)
            rr_sess = set(resp_sess) & set(zrr_sess)
            status_sess = occ_sess | rr_sess
        else:
            return None
        if None not in [fmcw, raw, cpx]:
            fmcw_cpx_stat = list(set(fmcw) & (set(cpx) - (set(raw) & set(cpx))) & status_sess)
            valid_setups = []
            for s in fmcw_cpx_stat:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and fw:
                    if nes_validity not in ['Invalid'] and (version.parse(fw) >= version.parse("0.4.7.5")):
                        valid_setups.append(s)
                else:
                    return None
            return sorted(valid_setups)
        return None

    def setup_status_fmcw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx or raw data
        (3) fw>='0.4.7.5' (4) EC=0&1 or ZRR=0&1
           :return: List of setup ids
           :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        raw = self.setup_by_vs(vs=VS.raw)
        cpx = self.setup_by_vs(vs=VS.cpx)
        full_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess = self.setup_vs_equals(VS.zrr, 1)
        resp_sess = self.setup_vs_equals(VS.zrr, 0)
        if None not in [full_sess, ec_sess, resp_sess, zrr_sess]:
            occ_sess = set(full_sess) & set(ec_sess)
            rr_sess = set(resp_sess) & set(zrr_sess)
            status_sess = occ_sess | rr_sess
        else:
            return None
        if None not in [fmcw, raw, cpx]:
            fmcw_raw_cpx = set(fmcw) & ((set(raw) | set(cpx)) - (set(raw) & set(cpx)))
            fmcw_raw_cpx_stat = list(fmcw_raw_cpx & status_sess)
            valid_setups = []
            for s in fmcw_raw_cpx_stat:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and fw:
                    if nes_validity not in ['Invalid'] and (version.parse(fw) >= version.parse("0.4.7.5")):
                        valid_setups.append(s)
                else:
                    return None
            return sorted(valid_setups)
        return None

    def setup_status_cw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) cw (1) nes validity is 'valid' or 'confirmed' (2) EC=0&1 or ZRR=0&1
           :return: List of setup ids
           :rtype: List[int]
        """
        cw = self.sr_cw_setups()
        full_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess = self.setup_vs_equals(VS.zrr, 1)
        resp_sess = self.setup_vs_equals(VS.zrr, 0)
        if None not in [full_sess, ec_sess, resp_sess, zrr_sess]:
            occ_sess = set(full_sess) & set(ec_sess)
            rr_sess = set(resp_sess) & set(zrr_sess)
            status_sess = occ_sess | rr_sess
        else:
            return None
        if None not in [cw]:
            cw_stat = list(set(cw) & status_sess)
            valid_setups = []
            for s in cw_stat:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                fw = self.setup_fw(s)
                if nes_validity and fw:
                    if nes_validity not in ['Invalid']:
                        valid_setups.append(s)
                else:
                    return None
            return sorted(valid_setups)
        return None

    def setup_for_test_fmcw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) fw >='0.4.7.5'
        (4) has reference HR (5) reference HR is 'valid' or 'confirmed' (6) continues reference
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        hr = self.setup_by_vs(vs=VS.hr)
        if None not in [fmcw, cpx, hr]:
            fmcw_cpx_hr = list(set(fmcw) & set(cpx) & set(hr))
            fmcw_cpx_hr = self._remove_raw_cpx(fmcw_cpx_hr)
            valid_setups = []
            for s in fmcw_cpx_hr:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                ref = self.setup_ext_proc_ref(setup=s)
                fw = self.setup_fw(s)
                if nes_validity and ref and fw:
                    for r in ref:
                        if r.lower() not in NONE_CONTINUES_SENSORS:
                            vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.hr)
                            if vs_validity:
                                if nes_validity not in ['Invalid'] and vs_validity not in ['Invalid'] and \
                                        (version.parse(fw) >= version.parse("0.4.7.5")):
                                    valid_setups.append(s)
                                    break
                else:
                    return None
            return valid_setups
        return None

    def setup_fmcw(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data (3) fw >='0.4.7.5'
        (4) has reference HR (5) reference HR is 'valid' or 'confirmed' (6) continues reference
        (7) excluding sessions containing 'Tidal' in Note (8) duration < 480 sec
                :return: List of setup ids
                :rtype: List[int]
        """
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        hr = self.setup_by_vs(vs=VS.hr)
        note = self.setup_by_note(note='Tidal')
        if None not in [fmcw, cpx, hr, note]:
            fmcw_cpx_hr = list((set(fmcw) & set(cpx) & set(hr)) - set(note))
            fmcw_cpx_hr = self._remove_raw_cpx(fmcw_cpx_hr)
            valid_setups = []
            for s in fmcw_cpx_hr:
                nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
                ref = self.setup_ext_proc_ref(setup=s)
                fw = self.setup_fw(s)
                if nes_validity and ref and fw:
                    for r in ref:
                        if r.lower() not in NONE_CONTINUES_SENSORS:
                            vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.hr)
                            duration = self.setup_duration(setup=s)
                            if duration:
                                if vs_validity:
                                    if nes_validity not in ['Invalid'] and vs_validity not in ['Invalid'] and \
                                            (version.parse(fw) >= version.parse("0.4.7.5")) and duration < 480:
                                        valid_setups.append(s)
                                        break
                            else:
                                return None
                else:
                    return None
            return valid_setups
        return None

    def setup_szmc(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) company = 'szmc' (1) ref (HR) positive values, (2) ref (RR) positive values
           :return: List of setup ids
           :rtype: List[int]
        """

        #  === szmc setups  === #
        szmc_sess = self.setup_by_project(prj=Project.szmc)

        #  === ref setups  === #
        hr_pos = self.setup_vs_greater_than(vs=VS.hr, value=0)
        rr_pos = self.setup_vs_greater_than(vs=VS.rr, value=0)
        hr_none_pos = self.setup_vs_smaller_than(vs=VS.hr, value=1)
        rr_none_pos = self.setup_vs_smaller_than(vs=VS.rr, value=1)

        if None not in [szmc_sess, hr_pos, rr_pos, hr_none_pos, rr_none_pos]:
            return sorted(list((set(szmc_sess) & set(hr_pos) & set(rr_pos)) - (set(hr_none_pos) | set(rr_none_pos))))
        else:
            return None

    def setup_fae_rest(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data
        (3) fw>='0.4.7.5' or '0.1.3.0' <= fw < '0.3.0.0'
        (4) EC=0, (5) ZRR=0, (6) Motion=0, (7) Speaking=0, (8) Driving=0, (9) 80<duration<600 sec,
        (10) ref (HR) validity is 'valid' or 'confirmed', (11) ref (RR) validity is 'valid' or 'confirmed',
         (12) excluding sessions containing 'Tidal' in Note, (13) env='lab' (14) fs=500Hz,
         (15) reference delay < 60s (16) ref (HR) positive values, (17) ref (RR) positive values,
         (18) ref=Mindray, (19) model=Myrtus, (20) company= neteera or szmc or moh or cenExel
           :return: List of setup ids
           :rtype: List[int]
        """
        # shawarma = set(self.setup_by_note('Data collection, lying on the bed - on the back side and belly'))

        #  === gt recorder  === #
        valid_setups = []
        full_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess = self.setup_vs_equals(VS.zrr, 1)
        resp_sess = self.setup_vs_equals(VS.zrr, 0)
        rest_sess = self.setup_vs_equals(VS.rest, 1)
        motion_sess = self.setup_vs_equals(VS.rest, 0)
        speak_sess = self.setup_vs_equals(VS.speaking, 1)
        no_speak_sess = self.setup_vs_equals(VS.speaking, 0)
        station_sess = self.setup_vs_equals(VS.stationary, 1)
        drive_sess = self.setup_vs_equals(VS.stationary, 0)

        #  === state  === #
        gt_recorder_sessions = set(self.setup_by_sensor(Sensor.gt_recorder))
        rest_sess_state = self.setup_by_state(state=State.is_rest, value=True)
        full_sess_state = self.setup_by_state(state=State.is_occupied, value=True)
        # none_motion_sess_state = set(self.setup_by_state(state=State.is_motion, value=False)) | shawarma
        none_motion_sess_state = self.setup_by_state(state=State.is_motion, value=False)
        none_speaking_sess_state = self.setup_by_state(state=State.is_speaking, value=False)
        none_hb_sess_state = self.setup_by_state(state=State.is_hb, value=False)
        none_ec_sess_state = self.setup_by_state(state=State.is_empty, value=False)
        none_drive_sess_state = self.setup_by_state(state=State.is_driving, value=False)
        none_drive_idle_state = self.setup_by_state(state=State.is_driving_idle, value=False)

        full_valid_sess = set(full_sess) - set(ec_sess)
        resp_valid_sess = set(resp_sess) - set(zrr_sess)
        # rest_valid_sess = (set(rest_sess) - set(motion_sess)) | shawarma
        rest_valid_sess = set(rest_sess) - set(motion_sess)
        no_speak_valid_sess = set(no_speak_sess) - set(speak_sess)
        station_valid_sess = set(station_sess) - set(drive_sess)

        gt_sess = full_valid_sess & resp_valid_sess & rest_valid_sess & no_speak_valid_sess & station_valid_sess
        state_sess = set(rest_sess_state) & set(full_sess_state) & set(none_motion_sess_state) & \
                     set(none_speaking_sess_state) & set(none_hb_sess_state) & set(none_ec_sess_state) & \
                     set(none_drive_sess_state) & set(none_drive_idle_state)
        gt_state_sess = (state_sess - gt_recorder_sessions) | (gt_sess & state_sess)
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        tidal = self.setup_by_note(note='Tidal')
        # hr_pos = self.setup_vs_greater_than(vs=VS.hr, value=0)
        # rr_pos = self.setup_vs_greater_than(vs=VS.rr, value=0)
        # hr_none_pos = self.setup_vs_smaller_than(vs=VS.hr, value=1)
        # rr_none_pos = self.setup_vs_smaller_than(vs=VS.rr, value=1)
        hr_pos = self.setup_vs_in_range(vs=VS.hr, minn=1, maxx=161)
        rr_pos = self.setup_vs_in_range(vs=VS.rr, minn=1, maxx=41)
        invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
        lab = self.setup_by_environment(Environment.lab)
        mindray = self.setup_by_sensor(sensor=Sensor.epm_10m)
        neteera = self.setup_by_project(prj=Project.neteera)
        szmc = self.setup_by_project(prj=Project.szmc)
        cen_exel = self.setup_by_project(prj=Project.cen_exel)
        moh = self.setup_by_project(prj=Project.moh)
        model = self.setup_by_model(model=Model.myrtus)
        stress = self.setup_by_note('stress')
        standing = self.setup_by_posture(Posture.standing)
        under_bed = self.setup_by_mount(Mount.bed_bottom)

        gt_fmcw_cpx_note_vs_lab = (gt_state_sess & set(fmcw) & set(cpx) & set(hr_pos) & set(rr_pos) &
                                   set(lab)) - set(tidal + invalid + stress + standing + under_bed)
        sess = self._remove_raw_cpx(gt_fmcw_cpx_note_vs_lab)
        sess = list(set(sess) & set(mindray) & set(model) & (set(neteera) | set(szmc) | set(moh) | set(cen_exel)))
        for s in sess:
            ref = self.setup_ext_proc_ref(setup=s)
            fs = self.setup_fs(s)
            for r in ref:
                if r.lower() not in NONE_CONTINUES_SENSORS and r.lower() not in GT_SENSORS:
                    sensor_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()])
                    hr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.hr)
                    rr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.rr)
                    duration = self.setup_duration(setup=s)
                    delay = self.setup_delay(setup=s, sensor=Sensor[r.lower()])
                    if delay is None:
                        delay = 0
                    if ('Invalid' not in [hr_vs_validity, rr_vs_validity, sensor_validity] and duration and 80 <
                            duration < 600 and fs == 500.0 and abs(delay) < 60):
                        valid_setups.append(s)
                        break
        return sorted(valid_setups)

    def setup_fae_rest_back_front(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data
        (3) fw>='0.4.7.5' or '0.1.3.0' <= fw < '0.3.0.0'
        (4) EC=0, (5) ZRR=0, (6) Motion=0, (7) Speaking=0, (8) Driving=0, (9) 80<duration<600 sec,
        (10) ref (HR) validity is 'valid' or 'confirmed', (11) ref (RR) validity is 'valid' or 'confirmed',
         (12) excluding sessions containing 'Tidal' in Note, (13) env='lab' (14) fs=500Hz,
         (15) reference delay < 60s,
         (16) target=back & mount=seat_back|bed_bottom or target=front & mount=lab_ceiling|tripod|bed_top,
         (17) ref (HR) positive values, (18) ref (RR) positive values, (19) ref=Mindray, (20), model=Myrtus,
         (21) company= neteera or szmc or moh
           :return: List of setup ids
           :rtype: List[int]
        """

        #  === gt recorder  === #
        valid_setups = []
        full_sess = self.setup_vs_equals(VS.occupancy, 1)
        ec_sess = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess = self.setup_vs_equals(VS.zrr, 1)
        resp_sess = self.setup_vs_equals(VS.zrr, 0)
        rest_sess = self.setup_vs_equals(VS.rest, 1)
        motion_sess = self.setup_vs_equals(VS.rest, 0)
        speak_sess = self.setup_vs_equals(VS.speaking, 1)
        no_speak_sess = self.setup_vs_equals(VS.speaking, 0)
        station_sess = self.setup_vs_equals(VS.stationary, 1)
        drive_sess = self.setup_vs_equals(VS.stationary, 0)

        #  === state  === #
        rest_sess_state = self.setup_by_state(state=State.is_rest, value=True)
        full_sess_state = self.setup_by_state(state=State.is_occupied, value=True)
        none_motion_sess_state = self.setup_by_state(state=State.is_motion, value=False)
        none_speaking_sess_state = self.setup_by_state(state=State.is_speaking, value=False)
        none_hb_sess_state = self.setup_by_state(state=State.is_hb, value=False)
        none_ec_sess_state = self.setup_by_state(state=State.is_empty, value=False)
        none_drive_sess_state = self.setup_by_state(state=State.is_driving, value=False)
        none_drive_idle_state = self.setup_by_state(state=State.is_driving_idle, value=False)

        back = self.setup_by_target(target=Target.back)
        front = self.setup_by_target(target=Target.front)

        seat_back = self.setup_by_mount(mount=Mount.seat_back)
        bed_bottom = self.setup_by_mount(mount=Mount.bed_bottom)
        lab_ceiling = self.setup_by_mount(mount=Mount.lab_ceiling)
        bed_top = self.setup_by_mount(mount=Mount.bed_top)
        tripod = self.setup_by_mount(mount=Mount.tripod)

        if None not in [full_sess, ec_sess, resp_sess, zrr_sess, rest_sess, motion_sess, speak_sess, no_speak_sess,
                        station_sess, drive_sess, rest_sess_state, full_sess_state, none_motion_sess_state,
                        none_speaking_sess_state, none_hb_sess_state, none_ec_sess_state, none_drive_sess_state,
                        none_drive_idle_state, seat_back, bed_bottom, lab_ceiling, bed_top, tripod]:
            full_inter_sess = set(full_sess) & set(ec_sess)
            resp_inter_sess = set(resp_sess) & set(zrr_sess)
            rest_inter_sess = set(rest_sess) & set(motion_sess)
            no_speak_inter_sess = set(no_speak_sess) & set(speak_sess)
            full_valid_sess = set(full_sess) - full_inter_sess
            resp_valid_sess = set(resp_sess) - resp_inter_sess
            rest_valid_sess = set(rest_sess) - rest_inter_sess
            no_speak_valid_sess = set(no_speak_sess) - no_speak_inter_sess
            station_valid_sess = set(station_sess) - set(drive_sess)

            gt_sess = full_valid_sess & resp_valid_sess & rest_valid_sess & no_speak_valid_sess & station_valid_sess
            state_sess = set(rest_sess_state) & set(full_sess_state) & set(none_motion_sess_state) & \
                         set(none_speaking_sess_state) & set(none_hb_sess_state) & set(none_ec_sess_state) & \
                         set(none_drive_sess_state) & set(none_drive_idle_state)
            gt_state_sess = gt_sess | state_sess
            sess = gt_state_sess & ((set(back) & (set(seat_back) | set(bed_bottom))) |
                                    (set(front) & (set(lab_ceiling) | set(bed_top) | set(tripod))))
            fmcw = self.sr_fmcw_setups()
            cpx = self.setup_by_vs(vs=VS.cpx)
            note = self.setup_by_note(note='Tidal')
            hr_pos = self.setup_vs_greater_than(vs=VS.hr, value=0)
            rr_pos = self.setup_vs_greater_than(vs=VS.rr, value=0)
            hr_none_pos = self.setup_vs_smaller_than(vs=VS.hr, value=1)
            rr_none_pos = self.setup_vs_smaller_than(vs=VS.rr, value=1)
            invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
            lab = self.setup_by_environment(Environment.lab)
            mindray = self.setup_by_sensor(sensor=Sensor.epm_10m)
            neteera = self.setup_by_project(prj=Project.neteera)
            szmc = self.setup_by_project(prj=Project.szmc)
            moh = self.setup_by_project(prj=Project.moh)
            model = self.setup_by_model(model=Model.myrtus)

            if None not in [fmcw, cpx, note, hr_pos, rr_pos, hr_none_pos, rr_none_pos, lab, invalid, mindray, neteera,
                            szmc, moh, model]:
                sess = list((sess & set(fmcw) & set(cpx) & set(hr_pos) & set(rr_pos) & set(lab)) -
                            (set(hr_none_pos) | set(rr_none_pos) | set(note) | set(invalid)))
                sess = self._remove_raw_cpx(sess)
                sess = list(set(sess) & set(mindray) & set(model) & (set(neteera) | set(szmc) | set(moh)))

                for s in sess:
                    ref = self.setup_ext_proc_ref(setup=s)
                    fw = self.setup_fw(s)
                    fs = self.setup_fs(s)
                    if ref and fw and fs:
                        for r in ref:
                            if r.lower() not in NONE_CONTINUES_SENSORS and r.lower() not in GT_SENSORS:
                                sensor_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()])
                                hr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.hr)
                                rr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.rr)
                                duration = self.setup_duration(setup=s)
                                delay = self.setup_delay(setup=s, sensor=Sensor[r.lower()])
                                if sensor_validity and duration:
                                    if (((hr_vs_validity and hr_vs_validity not in ['Invalid']) or not hr_vs_validity)
                                            and ((rr_vs_validity and rr_vs_validity not in [
                                                'Invalid']) or not rr_vs_validity)
                                            and sensor_validity not in ['Invalid']
                                            and ((version.parse(fw) >= version.parse("0.4.7.5")) or
                                                 (version.parse("0.1.3.1") <= version.parse(fw) < version.parse(
                                                     "0.3.4.1")))
                                            and 80 < duration < 600 and fs == 500.0 and (
                                                    (delay and abs(delay) < 60) or not delay)):
                                        valid_setups.append(s)
                                        break
                                else:
                                    return None
                    else:
                        return None
                return sorted(valid_setups)
            else:
                return None
        else:
            return None

    def setup_ec_benchmark(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data
        (3) fw>='0.4.7.5' or '0.1.3.0' <= fw < '0.3.0.0' (4) EC=1 | ZRR = 1 (5) env='lab', (6) fs=500Hz
           :return: List of setup ids
           :rtype: List[int]
        """
        valid_setups = []
        ec_sess_gt = self.setup_vs_equals(VS.occupancy, 0)
        zrr_sess_gt = self.setup_vs_equals(VS.zrr, 1)
        ec_sess = set(ec_sess_gt) | set(zrr_sess_gt)

        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        lab = self.setup_by_environment(Environment.lab)
        ec_fmcw_cpx_lab = list(set(ec_sess) & set(fmcw) & set(cpx) & set(lab))
        ec_fmcw_cpx_lab_sess = self._remove_raw_cpx(ec_fmcw_cpx_lab)

        for s in ec_fmcw_cpx_lab_sess:
            gt = self.setup_ref_path(setup=s, sensor=Sensor.gt_recorder)
            if not gt:
                continue
            ref_validity = self.setup_data_validity(setup=s, sensor=Sensor.gt_recorder)
            nes_validity = self.setup_data_validity(setup=s, sensor=Sensor.nes)
            fs = self.setup_fs(s)
            sn = self.setup_sn(s)[0]
            if 'Invalid' not in [nes_validity, ref_validity] and fs == 500.0 and sn != '131020450074':
                multi = self.setup_multi(s)
                if (len(multi) < 2 or
                        len({self.setup_radar_config(idx)['frontendConfig_baseFreq'] for idx in multi}) > 1):
                    valid_setups.append(s)
        return valid_setups

    def setup_nwh_benchmark(self):
        output_setups = []
        nwh_company = self.setup_by_project(Project.nwh)
        bed_top = self.setup_by_mount(Mount.bed_top)
        invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
        setups = set(bed_top) & set(nwh_company) - set(invalid) - \
                 {9101, 9187, 9002, 9096, 6283, 8579, 8584, 8724, 8734, 8756, 8919, 9094, 8705, 6641, 8994, 8999, 9050}
        for s in setups:
            natus_validity = self.setup_data_validity(s, Sensor.natus)
            if natus_validity in ['Confirmed', 'Valid']:
                output_setups.append(s)
        return output_setups

    def setup_med_bridge_benchmark(self):
        return self.setup_by_project('F1-N.Raleigh') +\
               self.setup_by_project('F2-Duraleigh') + self.setup_by_project('F3-Clayton')

    def setup_cen_exel_benchmark(self):
        output_setups = []
        nwh_company = self.setup_by_project('CenExel')
        invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
        setups = set(nwh_company) - set(invalid)
        for s in setups:
            subject = self.setup_subject(s)
            if subject == 'EXT' or 'CenExel' in subject:
                output_setups.append(s)
        return output_setups

    def setup_ie_benchmark(self):
        output_setups = []
        invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
        spirometer = self.setup_by_sensor(Sensor.spirometer)
        setups = set(spirometer) - set(invalid)
        for s in setups:
            natus_validity = self.setup_data_validity(s, Sensor.spirometer)
            if natus_validity in ['Confirmed', 'Valid']:
                output_setups.append(s)
        return output_setups

    def setup_mild_motion_benchmark(self) -> Optional[List[int]]:
        """ Retrieve setup ids for (0) fmcw (1) nes validity is 'valid' or 'confirmed' (2) cpx data
        (4) EC=0, (5) ZRR=0, (8) Driving=0, (9) duration>45 sec,
        (10) ref (HR) validity is 'valid' or 'confirmed', (11) ref (RR) validity is 'valid' or 'confirmed',
        (12) excluding sessions containing 'Tidal' in Note, (13) standing=0 (14) env='lab' (15) fs=500Hz,
        (16) reference delay < 60s, (17) scenario contains ’Mild motion’
           :return: List of setup ids
           :rtype: List[int]
        """
        valid_setups = []

        valid_distance = set(self.setup_distance_in_range(49, 1601))
        fmcw = self.sr_fmcw_setups()
        cpx = self.setup_by_vs(vs=VS.cpx)
        hr = self.setup_by_vs(vs=VS.hr)
        rr = self.setup_by_vs(vs=VS.rr)
        standing = self.setup_by_posture(posture=Posture.standing)
        invalid = self.setup_by_data_validity(sensor=Sensor.nes, value=Validation.invalid)
        lab = self.setup_by_environment(Environment.lab)
        mild_motion = self.setup_by_scenario('Mild motion')

        if None not in [fmcw, cpx, hr, rr, standing, lab, invalid, mild_motion]:
            gt_fmcw_cpx_note_vs_lab = (valid_distance & set(fmcw) & set(cpx) & set(mild_motion) & set(hr) & set(rr) &
                                       set(lab)) - (set(standing) | set(invalid))
            gt_fmcw_cpx_note_vs_lab_sess = self._remove_raw_cpx(gt_fmcw_cpx_note_vs_lab)
            for s in gt_fmcw_cpx_note_vs_lab_sess:
                ref = self.setup_ext_proc_ref(setup=s)
                fs = self.setup_fs(s)
                if ref and fs:
                    for r in ref:
                        if r.lower() not in NONE_CONTINUES_SENSORS and r.lower() not in GT_SENSORS:
                            hr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.hr)
                            rr_vs_validity = self.setup_data_validity(setup=s, sensor=Sensor[r.lower()], vs=VS.rr)
                            duration = self.setup_duration(setup=s)
                            delay = self.setup_delay(setup=s, sensor=Sensor[r.lower()])
                            if hr_vs_validity not in ['Invalid'] and rr_vs_validity not in ['Invalid'] \
                                    and duration > 45 and fs == 500.0 and (delay is None or abs(delay) < 60):
                                valid_setups.append(s)
                                break
                else:
                    return None
            return valid_setups
        else:
            return None

    def has_id_project(self, setup: int, prj: Project) -> bool:
        """Check if id project exists for a given setup

        :param int setup: Unique identifier for the setup
        :param Project prj: Project name
        :return: True/False
        :rtype: bool
        """
        try:

            self._execute("""SELECT join_setup_feature_param.* FROM join_setup_feature_param, 
            feature_param WHERE fk_setup_id = %s AND 
            join_setup_feature_param.fk_feature_param_id = feature_param.feature_param_id AND 
            feature_param.name = %s""", [setup, 'id_{}'.format(prj)])
            count = self.cursor.rowcount
            return True if count > 0 else False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return False

    def set_setup_project_id(self, setup: int, prj: Project, value: str) -> bool:
        """ insert new project id by setup_id and project name

                :param int setup: Unique identifier for the setup
                :param Project prj: project name
                :param value: id value
                :return: False/True
                :rtype: bool
                """
        setup_uuid = self._setup_uuid_by_id(setup)
        feature_uuid, feature_id = self._feature_param_ids('id_{}'.format(prj))
        if setup_uuid:
            if not self.has_id_project(setup, prj):
                return self._insert_join_setup_feature_param(setup_uuid, setup, feature_uuid, feature_id, value)
            else:
                return self.__update_join_setup_feature_param(setup_uuid, feature_uuid, value)
        return False

    def setup_by_project_id(self, prj: Project, pid: int) -> Optional[int]:
        """ Return setup id for a given project name and setup project id

        :param Project prj: project name
        :param int pid: setup project id
        :return: setup id
        :rtype: int
        """
        try:

            self._execute("""
            SELECT setup.setup_id
            FROM join_setup_feature_param, feature_param, feature_param_type, setup
            WHERE join_setup_feature_param.fk_setup_uuid = setup.setup_uuid
            AND join_setup_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid
            AND  feature_param.fk_feature_param_type_uuid = feature_param_type.feature_param_type_uuid
            AND  feature_param_type.name = 'customer_id'
            AND feature_param.name = %s
            AND join_setup_feature_param.value = %s""",
                          ['id_{}'.format(prj), str(pid)])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_version(self, version: str) -> List[int]:
        """ Retrieve setup ids for which version exist

            :param str version: version name
            :return: List of setup ids
            :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM test, setup, sw_version WHERE 
            test.fk_setup_uuid = setup.setup_uuid AND test.fk_sw_version_uuid = sw_version.sw_version_uuid AND 
            sw_version.name = %s ORDER BY setup.setup_id""", [version])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_by_model(self, model: Union[Model, str]) -> List[int]:
        """ Retrieve setup ids for which model exist

            :param str model: model name
            :return: List of setup ids
            :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM nes_model, setup, nes_config WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.fk_nes_model_uuid = nes_model.nes_model_uuid AND 
            nes_model.name LIKE %s ORDER BY setup.setup_id""", ["%" + str(model) + "%"])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_by_note(self, note: str) -> List[int]:
        """ Retrieve setup ids for which note exist

            :param str note: note to search
            :return: List of setup ids
            :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM session, setup WHERE 
            setup.fk_session_uuid = session.session_uuid AND session.note LIKE %s ORDER BY setup.setup_id""",
                          ["%" + note + "%"])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_continues_ref(self) -> List[int]:
        """ Retrieve cw setup ids for which continues reference exist (excluding 'elbit_ecg_sample')
            :return: List of setup ids
            :rtype: List[int]
        """
        cw_setups = set(self.sr_cw_setups())
        setups_with_ref = []
        if cw_setups:
            for s in cw_setups:
                ref = self.setup_ext_proc_ref(s)
                if ref and not (len(ref) == 1 and Sensor.elbit_ecg_sample.name.upper() in ref):
                    setups_with_ref.append(s)
        return sorted(setups_with_ref)

    def setup_subject_details(self, setup: int) -> Optional[Dict]:
        """ Return subject gender, birth year, height, weight by setup

            :param int setup: Unique identifier for the setup
            :return: subject details
            :rtype: dict
            """
        try:
            self._execute("""
            SELECT subject.gender, subject.birth_year, subject.height_cm, subject.weight_kg, subject.note
            FROM setup, session, subject
            WHERE  setup.fk_session_uuid = session.session_uuid
            AND session.fk_subject_uuid = subject.subject_uuid
            AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return {'gender': res[0][0], 'birthYear': res[0][1], 'height': res[0][2], 'weight': res[0][3],
                        'note': res[0][4]}
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_vs(self, vs: VS, sensor: Optional[Sensor] = None) -> Optional[List[int]]:
        """ A list of setup ids for which vs exist

        :param str vs: Unique identifier name for vs
        :param Sensor sensor: Optional sensor type
        :return: A list of setup ids for which vs exist
        :rtype: List[int]
        """
        try:
            if not sensor:
                sql = """
                SELECT DISTINCT T.SETUP FROM((
                    SELECT setup.setup_id AS SETUP
                    FROM setup, data, data_npy, feature_param
                    WHERE setup.setup_uuid = data.fk_setup_uuid
                    AND data_npy.fk_data_uuid = data.data_uuid
                    AND data_npy.fk_feature_param_uuid = feature_param.feature_param_uuid
                    AND feature_param.name = '{0}'
                    ORDER BY setup.setup_id) UNION ALL (SELECT setup.setup_id AS SETUP FROM data, data_npy, setup, 
                session, feature_param WHERE setup.fk_session_uuid = session.session_uuid AND 
                setup.fk_session_uuid = data.fk_session_uuid AND data.data_uuid = data_npy.fk_data_uuid AND 
                data_npy.fk_feature_param_uuid = feature_param.feature_param_uuid AND feature_param.name = '{0}'))T 
                ORDER BY T.SETUP""".format(vs)
            else:
                sql = """SELECT DISTINCT T.SETUP FROM(
                (SELECT setup.setup_id AS SETUP FROM setup, data, data_npy, sensor,
                feature_param WHERE setup.setup_uuid = data.fk_setup_uuid AND data_npy.fk_data_uuid = data.data_uuid AND 
                data_npy.fk_feature_param_uuid = feature_param.feature_param_uuid AND feature_param.name = '{0}' AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND sensor.name = '{1}' ORDER BY 
                setup.setup_id) UNION ALL (SELECT setup.setup_id AS SETUP FROM data, data_npy, setup, sensor, 
                session, feature_param WHERE setup.fk_session_uuid = session.session_uuid AND 
                setup.fk_session_uuid = data.fk_session_uuid AND data.data_uuid = data_npy.fk_data_uuid AND 
                data_npy.fk_feature_param_uuid = feature_param.feature_param_uuid AND feature_param.name = '{0}' AND 
                data.fk_sensor_uuid = sensor.sensor_uuid AND sensor.name = '{1}'))T 
                ORDER BY T.SETUP""".format(vs, sensor)
            self._execute(sql)
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def set_subject_gender(self, setup: int, gender: Gender) -> bool:
        """ set subject gender by setup id
                    :param int setup: Unique identifier for setup
                    :param Gender gender: subject gender
                    :return: False/True
                    :rtype: bool
                    """
        subject_id = self.__subject_id(setup)
        return self.__update_subject_gender(subject_id, gender)

    def __update_subject_gender(self, subject_id: int, gender: Gender) -> bool:
        """ update subject gender in db
                    :param int subject_id: Unique identifier of the subject
                    :param Gender gender: subject gender
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE subject SET gender = %s WHERE subject_id = %s""",
                                           [gender, subject_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_subject_age(self, setup: int, age: int) -> bool:
        """ set subject age by setup id
                    :param int setup: Unique identifier for setup
                    :param int age: subject age
                    :return: False/True
                    :rtype: bool
                    """
        assert 0 <= age <= 255, 'incorrect age'
        subject_id = self.__subject_id(setup)
        year = datetime.datetime.now().year
        return self.__update_subject_biy(subject_id, year - age)

    def __update_subject_biy(self, subject_id: int, biy: int) -> bool:
        """ update subject birth year in db
                    :param int subject_id: Unique identifier of the subject
                    :param int biy: subject birth year
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE subject SET birth_year = %s WHERE subject_id = %s""",
                                           [biy, subject_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_subject_height(self, setup: int, height: int) -> bool:
        """ set subject height by setup id
                    :param int setup: Unique identifier for setup
                    :param int height: subject height
                    :return: False/True
                    :rtype: bool
                    """
        assert 0 <= height <= 255, 'incorrect height'
        subject_id = self.__subject_id(setup)
        return self.__update_subject_height(subject_id, height)

    def __update_subject_height(self, subject_id: int, height: int) -> bool:
        """ update subject height in db
                    :param int subject_id: Unique identifier of the subject
                    :param int height: subject height
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE subject SET height_cm = %s WHERE subject_id = %s""",
                                           [height, subject_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_subject_weight(self, setup: int, weight: int) -> bool:
        """ set subject weight by setup id
                    :param int setup: Unique identifier for setup
                    :param int weight: subject weight
                    :return: False/True
                    :rtype: bool
                    """
        assert 0 <= weight <= 255, 'incorrect weight'
        subject_id = self.__subject_id(setup)
        return self.__update_subject_weight(subject_id, weight)

    def __update_subject_weight(self, subject_id: int, weight: int) -> bool:
        """ update subject weight in db
                    :param int subject_id: Unique identifier of the subject
                    :param int weight: subject weight
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE subject SET weight_kg = %s WHERE subject_id = %s""",
                                           [weight, subject_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def setup_spot_res(self, setup: int) -> Optional[Dict[str, float]]:
        """ Retrieve Spot values by setup

        :param int setup: Unique identifier for setup
        :return: spot values as dict
        :rtype: Dict[str, float]
        """
        try:

            self._execute("""SELECT feature_param.name, join_test_feature_param.value 
            FROM setup, test, feature_param, feature_param_type, join_test_feature_param 
            WHERE setup.setup_uuid = test.fk_setup_uuid AND 
            join_test_feature_param.fk_test_uuid = test.test_uuid AND 
            join_test_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
            feature_param.fk_feature_param_type_uuid = feature_param_type.feature_param_type_uuid AND 
            feature_param_type.name = 'spot_statistics' AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            return {row[0]: row[1] for row in res if row[0] and row[1]}
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_radar_config(self, setup: int) -> Optional[Dict[str, str]]:
        """ Retrieve radar config dict for a given setup id

        :param int setup: Unique identifier for the setup
        :return: radar config dict
        :rtype: Dict[str, str]
        """
        try:

            self._execute("""SELECT nes_param.name, join_nes_config_param.value FROM setup, nes_config, 
            join_nes_config_param, nes_param, nes_param_type WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = join_nes_config_param.fk_nes_config_uuid AND 
            join_nes_config_param.fk_nes_param_uuid = nes_param.nes_param_uuid AND 
            nes_param.fk_nes_param_type_uuid = nes_param_type.nes_param_type_uuid AND setup.setup_id = %s AND 
            nes_param_type.name = 'radar_config' ORDER BY nes_param.name""", [setup])
            res = self.cursor.fetchall()
            return {row[0]: row[1] for row in res}
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_ADC_clkDiv(self, setup: int) -> Optional[str]:
        """ return 'ADC_clkDiv' param from raw nes data file for a given setup id

                :param int setup: setup id
                :return: ADC_clkDiv[MS/s]
                :rtype: str
                """
        pth = self.setup_ref_path(setup=setup, sensor=Sensor.nes)
        search = 'ADC_clkDiv[MS/s]='
        if pth:
            for p in pth:
                with open(p, 'r') as file:
                    for line in file:
                        if search in line:
                            search = search.replace('[', '\[')
                            l = re.search(search + '(.*),(.*)', line)
                            if l is not None:
                                return l.group(1).split(',')[0].replace(' ', '')
                            else:
                                l = re.search(search + '(.*)', line)
                                if l is not None:
                                    return l.group(1).replace(' ', '')
        return None

    def update_data_path(self, setup_id: int, old_path: str, new_path: str) -> bool:
        """ update benchmark-setup in db

                    :param int setup_id: Unique identifier for the setup
                    :param str old_path: path to be updated
                    :param str new_path: new path
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE data SET fpath = %s WHERE fpath = %s AND fk_setup_id = %s""",
                                           [new_path, old_path, setup_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                print('affected_count: {}'.format(affected_count))
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def setup_multi(self, setup: int) -> Optional[List[int]]:
        """ A list of setup ids of a multi session by setup id

        :param int setup: Unique identifier setup
        :return: List of setup ids
        :rtype: List[int]
        """
        try:
            self._execute(f"""
            SELECT se2.setup_id
            FROM setup se1, setup se2, session ss
            WHERE se1.fk_session_uuid = ss.session_uuid
            AND se2.fk_session_uuid = ss.session_uuid
            AND se1.setup_id= '{setup}'
            ORDER BY se2.setup_id""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def session_from_setup(self, setup_id: int) -> Optional[int]:
        """return the session number for a given setup"""
        return self._execute_and_fetch(f"""
            SELECT fk_session_id
            FROM setup
            WHERE setup_id = '{setup_id}'""")

    def setups_by_session(self, session_id: int) -> Optional[int]:
        """return the setup numbers for a given session"""
        try:
            self._execute(f"""
            SELECT setup_id
            FROM setup
            WHERE fk_session_id = '{session_id}'""")
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_from_to_date(self, from_date: Union[str, datetime.datetime], to_date: Union[str, datetime.datetime]=None)\
            -> Optional[List[int]]:
        """ Return all setups starting between required dates

                :param str from_date: date (strptime)
                :param str to_date: date (strptime)
                :return: List of setup ids
                :rtype: List[int]
                """
        if to_date is None:
            to_date = datetime.datetime.now()
        try:

            self._execute("""SELECT setup.setup_id FROM setup, setup_type, session WHERE
            setup.fk_setup_type_uuid = setup_type.setup_type_uuid AND 
            (setup_type.name='VS-HC' OR setup_type.name='VS-AUTO' OR setup_type.name='R&D') AND 
            setup.fk_session_uuid = session.session_uuid AND session.date BETWEEN %s AND %s""", [from_date, to_date])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_spot(self, setup: int) -> Optional[Dict]:
        """ retrieve spot params [spot type, spot t0, spot X duration] for a given setup ID
         :param int setup: Unique identifier setup
        :return: Dict spot params
        :rtype: Dict
        """
        try:

            self._execute("""SELECT nes_param_dynamic_value.value FROM setup, nes_param_dynamic, 
            nes_param_dynamic_value, nes_param WHERE nes_param_dynamic.fk_setup_uuid = setup.setup_uuid AND 
            nes_param_dynamic_value.fk_nes_param_dynamic_uuid = nes_param_dynamic.nes_param_dynamic_uuid AND 
            nes_param_dynamic.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'vsms_t0' AND 
            setup.setup_id = %s ORDER BY nes_param_dynamic_value.fk_nes_param_id""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return {'type': res[0][0], 't0': int(res[1][0]), 'X': int(res[2][0])}
            return {}
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_bin(self, setup: int) -> Optional[Dict]:
        """ retrieve bin params [range selected, bin selected] for a given setup ID
         :param int setup: Unique identifier setup
        :return: Dict bin params
        :rtype: Dict
        """
        try:

            self._execute("""SELECT nes_param_dynamic_value.value FROM setup, nes_param_dynamic, 
            nes_param_dynamic_value, nes_param WHERE nes_param_dynamic.fk_setup_uuid = setup.setup_uuid AND 
            nes_param_dynamic_value.fk_nes_param_dynamic_uuid = nes_param_dynamic.nes_param_dynamic_uuid AND 
            nes_param_dynamic.fk_nes_param_uuid = nes_param.nes_param_uuid AND nes_param.name = 'vsms_trace' AND 
            setup.setup_id = %s ORDER BY nes_param_dynamic_value.fk_nes_param_id""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return {'range': res[0][0], 'bin': int(res[1][0])}
            return {}
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __subject_ids(self, subject: str) -> tuple:
        """ Retrieve the subject ids by name

        :param str subject: subject name
        :return: subject_uuid: subject ids
        :rtype: str
        """
        try:

            self._execute("""SELECT subject_uuid, subject_id FROM subject WHERE subject.name = %s""", [subject])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __target_ids(self, target: Target) -> tuple:
        """ Retrieve the target ids by name

        :param Target target: target name
        :return: target_uuid, target_id: target ids
        :rtype: str
        """
        try:

            self._execute("""SELECT nes_subject_position_uuid, nes_subject_position_id FROM nes_subject_position WHERE 
            name = %s""", [target])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __mount_ids(self, mount: Mount) -> tuple:
        """ Retrieve the mount ids by name

        :param Mount mount: mount name
        :return: mount_uuid, mount_id: mount ids
        :rtype: str
        """
        try:

            self._execute("""SELECT nes_mount_area_uuid, nes_mount_area_id FROM nes_mount_area WHERE 
            name = %s""", [mount])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __component_inventory_ids(self, sn: str) -> tuple:
        """ Retrieve the sn ids

        :param str sn: s/n
        :return: sn_uuid, sn_id: sn ids
        :rtype: str
        """
        try:

            self._execute("""SELECT component_inventory_uuid, component_inventory_id FROM component_inventory WHERE 
            serial_num = %s""", [sn])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def set_setup_subject(self, setup: int, subject: str) -> bool:
        """ set subject name by setup id
                    :param int setup: Unique identifier for setup
                    :param str subject: subject name
                    :return: False/True
                    :rtype: bool
                    """
        session_uuid = self.__session_uuid(setup)
        subject_uuid, subject_id = self.__subject_ids(subject)
        if session_uuid and subject_uuid:
            return self.__update_subject_name(session_uuid, subject_uuid, subject_id)
        return False

    def __update_subject_name(self, session_uuid: str, subject_uuid: str, subject_id: int) -> bool:
        """ update subject session in db
                    :param str session_uuid: UUID of the session
                    :param str subject_uuid: UUID of the subject
                    :param int subject_id: Unique identifier subject
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE session SET fk_subject_uuid = %s, fk_subject_id = %s WHERE 
            session_uuid = %s""", [subject_uuid, subject_id, session_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def dbview_vsm_query(self) -> Optional[dict]:
        """ dbview vsm query (php page)
        :return: dbview vsm columns
        :rtype: dict
        """
        with open('dbview_vsm_query.txt', 'r') as f:
            output = f.read()
        try:

            self._execute("""{}""".format(output))
            res = self.cursor.fetchall()
            return res
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_sw_version(self, setup: int) -> Optional[str]:
        """ Retrieve sw version for a given setup id
            :param int setup: Unique identifier setup
            :return: sw version
            :rtype: str
        """
        try:

            self._execute("""SELECT sw_version.name FROM test, setup, sw_version WHERE 
            test.fk_setup_uuid = setup.setup_uuid AND test.fk_sw_version_uuid = sw_version.sw_version_uuid AND 
            setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return ''
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_sn(self, setup: int) -> Optional[List[str]]:
        """ Retrieve sn for a given setup id
            :param int setup: Unique identifier setup
            :return: sn
            :rtype: str
        """
        try:

            self._execute("""SELECT component_inventory.serial_num FROM nes_config, 
            join_nes_config_component_inventory, setup, component_inventory WHERE 
            setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            join_nes_config_component_inventory.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            join_nes_config_component_inventory.fk_component_inventory_uuid = component_inventory.component_inventory_uuid 
            AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_bins_smaller_than(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which number of bins smaller than <value> for cpx data only
        :param int value: max bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_bins, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param bins, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_bins.fk_nes_config_uuid AND 
            jnc_bins.fk_nes_param_uuid = bins.nes_param_uuid AND 
            bins.name = 'systemConfig_maxBinsToStream' AND jnc_bins.value < %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_bins_greater_than(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which number of bins greater than <value> for cpx data only
        :param int value: min bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_bins, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param bins, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_bins.fk_nes_config_uuid AND 
            jnc_bins.fk_nes_param_uuid = bins.nes_param_uuid AND 
            bins.name = 'systemConfig_maxBinsToStream' AND jnc_bins.value > %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_bins_equals(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which number of bins equals <value> for cpx data only
        :param int value: bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_bins, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param bins, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_bins.fk_nes_config_uuid AND 
            jnc_bins.fk_nes_param_uuid = bins.nes_param_uuid AND 
            bins.name = 'systemConfig_maxBinsToStream' AND jnc_bins.value = %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_bins_in_range(self, minn: int, maxx: int) -> Optional[List[int]]:
        """ retrieve setup ids for which number of bins between <minn> and <maxx> for cpx data only
        :param int minn: min bins value
        :param int maxx: max bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_bins, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param bins, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_bins.fk_nes_config_uuid AND 
            jnc_bins.fk_nes_param_uuid = bins.nes_param_uuid AND 
            bins.name = 'systemConfig_maxBinsToStream' AND (jnc_bins.value > %s AND jnc_bins.value < %s) AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [minn, maxx])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_offset_smaller_than(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which offset bin smaller than <value> for cpx data only
        :param int value: max bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_offset, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param offset, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_offset.fk_nes_config_uuid AND 
            jnc_offset.fk_nes_param_uuid = offset.nes_param_uuid AND 
            offset.name = 'systemConfig_binToStreamOffset' AND jnc_offset.value < %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_offset_greater_than(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which offset bin greater than <value> for cpx data only
        :param int value: min bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_offset, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param offset, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_offset.fk_nes_config_uuid AND 
            jnc_offset.fk_nes_param_uuid = offset.nes_param_uuid AND 
            offset.name = 'systemConfig_binToStreamOffset' AND jnc_offset.value > %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_offset_equals(self, value: int) -> Optional[List[int]]:
        """ retrieve setup ids for which offset bin equals <value> for cpx data only
        :param int value: bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_offset, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param offset, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_offset.fk_nes_config_uuid AND 
            jnc_offset.fk_nes_param_uuid = offset.nes_param_uuid AND 
            offset.name = 'systemConfig_binToStreamOffset' AND jnc_offset.value = %s AND 
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [value])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_offset_in_range(self, minn: int, maxx: int) -> Optional[List[int]]:
        """ retrieve setup ids for which offset bin between <minn> and <maxx> for cpx data only
        :param int minn: min bins value
        :param int maxx: max bins value
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM setup, nes_config, join_nes_config_param jnc_offset, 
            join_nes_config_param jnc_raw, join_nes_config_param jnc_cpx, nes_param offset, nes_param raw, nes_param cpx 
            WHERE setup.fk_nes_config_uuid = nes_config.nes_config_uuid AND 
            nes_config.nes_config_uuid = jnc_offset.fk_nes_config_uuid AND 
            jnc_offset.fk_nes_param_uuid = offset.nes_param_uuid AND 
            offset.name = 'systemConfig_binToStreamOffset' AND (jnc_offset.value > %s AND jnc_offset.value < %s) AND
            nes_config.nes_config_uuid = jnc_raw.fk_nes_config_uuid AND 
            jnc_raw.fk_nes_param_uuid = raw.nes_param_uuid AND raw.name = 'systemConfig_extendedFormat_rawData' 
            AND jnc_raw.value = '0' AND nes_config.nes_config_uuid = jnc_cpx.fk_nes_config_uuid AND 
            jnc_cpx.fk_nes_param_uuid = cpx.nes_param_uuid AND cpx.name = 'systemConfig_dataFramesEnable_raw' 
            AND jnc_cpx.value = '0' ORDER BY setup.setup_id""", [minn, maxx])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_distance_smaller_than(self, distance: int) -> Optional[List[int]]:
        """ retrieve setup ids for which distance is smaller than <distance in mm>
        :param int distance: distance in mm
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup_id FROM setup WHERE target_distance_mm < %s ORDER BY setup.setup_id""",
                          [distance])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_distance_greater_than(self, distance: int) -> Optional[List[int]]:
        """ retrieve setup ids for which distance is greater than <distance in mm>
         :param int distance: distance in mm
         :return: List of setup ids
         :rtype: List[int]
         """
        try:

            self._execute("""SELECT setup_id FROM setup WHERE target_distance_mm > %s ORDER BY setup.setup_id""",
                          [distance])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_distance_equals(self, distance: int) -> Optional[List[int]]:
        """ retrieve setup ids for which distance equals <distance in mm>
          :param int distance: distance in mm
          :return: List of setup ids
          :rtype: List[int]
          """
        try:

            self._execute("""SELECT setup_id FROM setup WHERE target_distance_mm = %s ORDER BY setup.setup_id""",
                          [distance])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_distance_in_range(self, minn: int, maxx: int) -> Optional[List[int]]:
        """ retrieve setup ids for which distance between <minn> and <maxx>
        :param int minn: min distace in mm
        :param int maxx: max distace in mm
        :return: List of setup ids
        :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup_id FROM setup WHERE 
            (target_distance_mm > %s AND target_distance_mm < %s) ORDER BY setup.setup_id""", [minn, maxx])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_env(self, setup: int) -> Optional[str]:
        """ Retrieve session environment for a given setup id
            :param int setup: Unique identifier setup
            :return: environment
            :rtype: str
        """
        try:

            self._execute("""SELECT environment_type.name FROM session, setup, environment_type, environment WHERE 
            setup.fk_session_uuid = session.session_uuid AND session.fk_environment_uuid = environment.environment_uuid 
            AND environment.fk_environment_type_uuid = environment_type.environment_type_uuid AND 
            setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            return res[0][0]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __setup_env_uuid(self, setup: int) -> Optional[str]:
        """ Retrieve session environment UUID for a given setup id
            :param int setup: Unique identifier setup
            :return: env_uuid
            :rtype: str
        """
        try:

            self._execute("""SELECT environment.environment_uuid FROM session, setup, environment WHERE 
            setup.fk_session_uuid = session.session_uuid AND session.fk_environment_uuid = environment.environment_uuid 
            AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            return res[0][0]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __seat(self, env_uuid: str) -> Optional[str]:
        """ Retrieve seat type for a given environment UUID
            :param str env_uuid: environment UUID
            :return: seat type
            :rtype: str
        """
        try:

            self._execute("""SELECT seat.name FROM seat, environment_lab WHERE 
            environment_lab.fk_seat_uuid = seat.seat_uuid AND environment_lab.fk_environment_uuid = %s""", [env_uuid])
            res = self.cursor.fetchall()
            return res[0][0]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __vehicle(self, env_uuid: str) -> Optional[str]:
        """ Retrieve vehicle type for a given environment UUID
            :param str env_uuid: environment UUID
            :return: vehicle type
            :rtype: str
        """
        try:

            self._execute("""SELECT vehicle_model.name FROM vehicle_model, vehicle, environment_vehicle WHERE 
            environment_vehicle.fk_vehicle_uuid = vehicle.vehicle_uuid AND 
            vehicle.fk_vehicle_model_uuid = vehicle_model.vehicle_model_uuid AND 
            environment_vehicle.fk_environment_uuid = %s""", [env_uuid])
            res = self.cursor.fetchall()
            return res[0][0]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_seat(self, setup: int) -> Optional[str]:
        """ Retrieve seat type for a given setup id
            :param int setup: Unique identifier setup
            :return: seat type
            :rtype: str
        """
        env = self.setup_env(setup)
        env_uuid = self.__setup_env_uuid(setup)
        if env and env_uuid:
            if env == 'Lab':
                return self.__seat(env_uuid)
            else:
                return self.__vehicle(env_uuid)
        else:
            return None

    def setup_posture(self, setup: int) -> Optional[str]:
        """ Get subject posture for a given setup id

        :param int setup: Unique identifier for the setup
        :return: posture of subject
        :rtype: str
        """
        try:

            self._execute("""SELECT subject_posture.name FROM setup, session, subject_posture WHERE 
            setup.fk_session_uuid = session.session_uuid AND 
            session.fk_subject_posture_uuid = subject_posture.subject_posture_uuid AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0]
            return None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def set_setup_target(self, setup: int, target: Target) -> bool:
        """ set target by setup id
                    :param int setup: Unique identifier for setup
                    :param Target target: target name
                    :return: False/True
                    :rtype: bool
                    """
        setup_uuid = self._setup_uuid_by_id(setup)
        target_uuid, target_id = self.__target_ids(target)
        if setup_uuid and target_uuid:
            return self.__update_taget(setup_uuid, target_uuid, target_id)
        return False

    def __update_taget(self, setup_uuid: str, target_uuid: str, target_id: int) -> bool:
        """ update target setup in db
                    :param str setup_uuid: UUID of the setup
                    :param str target_uuid: UUID of the target
                    :param int target_id: Unique identifier target
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE setup SET fk_nes_subject_position_uuid = %s, 
            fk_nes_subject_position_id = %s WHERE setup_uuid = %s""", [target_uuid, target_id, setup_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_setup_mount(self, setup: int, mount: Mount) -> bool:
        """ set mount by setup id
                    :param int setup: Unique identifier for setup
                    :param Mount mount: mount name
                    :return: False/True
                    :rtype: bool
                    """
        setup_uuid = self._setup_uuid_by_id(setup)
        mount_uuid, mount_id = self.__mount_ids(mount)
        if setup_uuid and mount_uuid:
            return self.__update_mount(setup_uuid, mount_uuid, mount_id)
        return False

    def __update_mount(self, setup_uuid: str, mount_uuid: str, mount_id: int) -> bool:
        """ update mount setup in db
                    :param str setup_uuid: UUID of the setup
                    :param str mount_uuid: UUID of the target
                    :param int mount_id: Unique identifier target
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE setup SET fk_nes_mount_area_uuid = %s, 
            fk_nes_mount_area_id = %s WHERE setup_uuid = %s""", [mount_uuid, mount_id, setup_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_setup_sn(self, setup: int, sn: str) -> bool:
        """ set s/n for a given setup id (supports 'Myrtus' HW and above)
                    :param int setup: Unique identifier for setup
                    :param str sn: s/n
                    :return: False/True
                    :rtype: bool
                    """
        sn_uuid, sn_id = self.__component_inventory_ids(sn)
        if sn_uuid:
            return self.__update_sn(setup, sn_uuid, sn_id)
        return False

    def __update_sn(self, setup_id: int, sn_uuid: str, sn_id: int) -> bool:
        """ update mount setup in db
                    :param int setup_id: Unique identifier setup
                    :param str sn_uuid: UUID of the sn
                    :param int sn_id: Unique identifier sn
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE join_nes_config_component_inventory 
            INNER JOIN nes_config ON join_nes_config_component_inventory.fk_nes_config_uuid = nes_config.nes_config_uuid 
            INNER JOIN setup ON nes_config.nes_config_uuid = setup.fk_nes_config_uuid AND setup.setup_id = %s 
            SET join_nes_config_component_inventory.fk_component_inventory_uuid = %s, 
            join_nes_config_component_inventory.fk_component_inventory_id = %s""", [setup_id, sn_uuid, sn_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def remove_setup_scenario(self, setup: int, scenario: str) -> Optional[bool]:
        """ Remove scenario from session given setup id and scenario name

            :param int setup: Unique identifier for setup
            :param str scenario: scenario to remove
            :return: False/True
            :rtype: bool
        """
        try:
            affected_count = self._execute("""DELETE jss.* FROM join_session_scenario jss
            INNER JOIN session ss ON jss.fk_session_uuid = ss.session_uuid
            INNER JOIN scenario sc ON jss.fk_scenario_uuid = sc.scenario_uuid
            INNER JOIN setup se ON ss.session_uuid = se.fk_session_uuid
            WHERE se.setup_id = %s AND sc.name LIKE %s""", [setup, scenario])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def insert_setup_scenario(self, setup: int, scenario: str) -> Optional[bool]:
        """ Insert scenario session given setup id and scenario name

            :param int setup: Unique identifier for setup
            :param str scenario: scenario to insert
            :return: False/True
            :rtype: bool
        """
        try:
            affected_count = self._execute("""INSERT INTO join_session_scenario (fk_session_uuid, fk_session_id, 
            fk_scenario_uuid, fk_scenario_id) SELECT ss.session_uuid, ss.session_id, sc.scenario_uuid, 
            sc.scenario_id FROM session ss, scenario sc, setup se WHERE ss.session_uuid = se.fk_session_uuid AND 
            se.setup_id = %s AND sc.name LIKE %s""", [setup, scenario])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_clk(self, sensor: Sensor) -> Optional[List[int]]:
        """ Retrieve setup ids for which reference and clk sync exists

        :param Sensor sensor: The type of sensor for which the clk sync exists
        :return: List of setup ids
        :rtype: List
        """
        try:

            self._execute("""SELECT setup.setup_id FROM data, sensor, setup WHERE 
            data.fk_session_uuid = setup.fk_session_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND
            sensor.name = %s AND data.clk_sync = 1 ORDER BY setup.setup_id""", [sensor])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_delay(self, sensor: Sensor) -> Optional[List[int]]:
        """ Retrieve setup ids for which nes-reference delay exists

        :param Sensor sensor: The type of sensor for which the delay exists
        :return: List of setup ids
        :rtype: List
        """
        try:

            self._execute("""SELECT setup.setup_id FROM data, sensor, setup, join_setup_data_param, feature_param WHERE 
            data.fk_session_uuid = setup.fk_session_uuid AND 
            data.fk_sensor_uuid = sensor.sensor_uuid AND
            join_setup_data_param.fk_setup_uuid = setup.setup_uuid AND 
            join_setup_data_param.fk_data_uuid = data.data_uuid AND
            join_setup_data_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND 
            feature_param.name = 'nes_ref_delay' AND sensor.name = %s ORDER BY setup.setup_id""", [sensor])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def vs_by_sensor(self, sensor: Sensor) -> Optional[List[str]]:
        """ Retrieve supported vs or a given reference

        :param Sensor sensor: The type of sensor
        :return: List of vs
        :rtype: List
        """
        try:

            self._execute("""SELECT feature_param.name FROM join_sensor_feature_param, sensor, feature_param WHERE 
            join_sensor_feature_param.fk_sensor_uuid = sensor.sensor_uuid AND 
            join_sensor_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND
            sensor.name = %s ORDER BY feature_param.name""", [sensor])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def sensor_by_vs(self, vs: VS) -> Optional[List[str]]:
        """ Retrieve supported sensors or a given vs

        :param VS vs: vs name
        :return: List of vs
        :rtype: List
        """
        try:

            self._execute("""SELECT sensor.name FROM join_sensor_feature_param, sensor, feature_param WHERE 
            join_sensor_feature_param.fk_sensor_uuid = sensor.sensor_uuid AND 
            join_sensor_feature_param.fk_feature_param_uuid = feature_param.feature_param_uuid AND
            feature_param.name = %s ORDER BY sensor.name""", [vs])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def __posture_ids(self, posture: Posture) -> tuple:
        """ get subject posture ids

                :param Posture posture: posture name
                :return: Posture ids
                :rtype: tuple
                """
        try:
            affected_count = self._execute("""SELECT subject_posture_uuid, subject_posture_id FROM subject_posture 
            WHERE name = %s""", [posture])
            res = self.cursor.fetchall()
            if affected_count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __update_posture(self, setup: int, posture_uuid: str, posture_id: int) -> Optional[bool]:
        """ update subject posture

                :param int setup: Unique identifier for the setup
                :param str posture_uuid: Posture UUID
                return: False/True
                :rtype: bool
                """
        try:
            affected_count = self._execute("""UPDATE session INNER JOIN setup ON 
            session.session_uuid = setup.fk_session_uuid AND setup.setup_id = %s SET 
            session.fk_subject_posture_uuid = %s, session.fk_subject_posture_id = %s""",
                                           [setup, posture_uuid, posture_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_posture(self, setup: int, posture: Posture) -> bool:
        """ set subject posture for a given setup id

            :param int setup: Unique identifier for the setup
            :param Posture posture: Posture type
            :return: False/True
            :rtype: bool
            """
        posture_uuid, posture_id = self.__posture_ids(posture)
        if posture_uuid:
            return self.__update_posture(setup, posture_uuid, posture_id)
        return False

    def set_distance(self, setup: int, distance: int) -> bool:
        """ set setup distance by setup id

            :param int setup: Unique identifier for the setup
            :param int distance: setup distance to be modified
            :return: False/True
            :rtype: bool
            """
        try:
            affected_count = self._execute("""UPDATE setup SET target_distance_mm = %s 
                WHERE setup.setup_id = %s""", [distance, setup])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def __nes_model_ids(self, model: str) -> tuple:
        """ Retrieve the NES model ids

        :param str model: model
        :return: model_uuid, model_id: model ids
        :rtype: str
        """
        try:

            self._execute("""SELECT nes_model_uuid, nes_model_id FROM nes_model WHERE name = %s""", [model])
            res = self.cursor.fetchall()
            count = self.cursor.rowcount
            if count > 0:
                return res[0][0], res[0][1]
            return None, None
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None, None

    def __update_nes_model(self, setup_id: int, model_uuid: str, model_id: int) -> bool:
        """ update model setup in db
                    :param int setup_id: Unique identifier setup
                    :param str model_uuid: UUID of the model
                    :param int model_id: Unique identifier model
                    :return: False/True
                    :rtype: bool
                    """
        try:
            affected_count = self._execute("""UPDATE nes_config 
            INNER JOIN nes_model ON nes_config.fk_nes_model_uuid = nes_model.nes_model_uuid 
            INNER JOIN setup ON nes_config.nes_config_uuid = setup.fk_nes_config_uuid AND setup.setup_id = %s 
            SET nes_config.fk_nes_model_uuid = %s, nes_config.fk_nes_model_id = %s""", [setup_id, model_uuid, model_id])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            if self.conn.rollback():
                self.conn.rollback()
            return False

    def set_model(self, setup: int, model: str) -> bool:
        """ set model for a given setup id
                    :param int setup: Unique identifier for setup
                    :param str model: model type
                    :return: False/True
                    :rtype: bool
                    """
        model_uuid, model_id = self.__nes_model_ids(model)
        if model_uuid:
            return self.__update_nes_model(setup, model_uuid, model_id)
        return False

    def setup_view(self, setup: Optional[int] = None) -> Optional[List[dict]]:
        """ get setup fields according to the db view page:
        Setup, Timestamp, Duration, Subject, Scenario, Path, Location, Target, Posture, State, Operator, Mode, FW,
        Model, Distance, SN, Version, Company, Note, Validity
                :param int setup: setup id
                :return: setup fields
                :rtype: dict
                """
        try:
            if setup:
                ss = " WHERE se.setup_id = {}".format(setup)
                se = " AND se.setup_id = {}".format(setup)
            else:
                ss = ''
                se = ''
            query = """SELECT Session, Setup, Timestamp, Duration, Subject, Gender, Scenario, Path, Location, Target, Posture, 
            State, CONCAT_WS(', ', GT_POS, GT_NEG) AS GT, Operator, Mode, FW, Model, Distance, SN, Version, Company, 
            Notes, Validity, Legacy


FROM session ss


  INNER JOIN
    ( SELECT ss.session_id AS Session,
             GROUP_CONCAT(DISTINCT se.setup_id ORDER BY se.setup_id SEPARATOR ', ') AS Setup,
             GROUP_CONCAT(DISTINCT se.legacy_id ORDER BY se.setup_id SEPARATOR ', ') AS Legacy,
             ss.date AS Timestamp,
             ss.duration_sec AS Duration,
             GROUP_CONCAT(DISTINCT sc.name ORDER BY sc.name SEPARATOR ', ') AS Scenario,
             CONCAT('{2}', DATE_FORMAT(date(ss.date), '%Y\%c\%e'), '//', se.fk_session_id) AS Path,
             su.name AS Subject,
             su.gender AS Gender,
             sp.name AS Posture,
             su2.name AS Operator,
             c.name AS Company,
             ss.note AS Notes
      FROM session ss
        JOIN setup se
          ON ss.session_id = se.fk_session_id
        LEFT JOIN subject su
          ON ss.fk_subject_id = su.subject_id
        LEFT JOIN join_session_scenario jss
          ON ss.session_id = jss.fk_session_id
        LEFT JOIN scenario sc
          ON jss.fk_scenario_id = sc.scenario_id
        LEFT JOIN neteera_staff_operator op
          ON ss.fk_neteera_staff_operator_id = op.neteera_staff_operator_id
        LEFT JOIN subject su2
          ON op.fk_subject_id = su2.subject_id
        LEFT JOIN subject_posture sp
          ON ss.fk_subject_posture_id = sp.subject_posture_id
        LEFT JOIN company c
          ON ss.fk_company_id = c.company_id 
      {0}
      GROUP BY session_id
    ) AS SESSION_DATA
        ON ss.session_id = SESSION_DATA.Session


  LEFT JOIN
          (SELECT ss.session_id,
           GROUP_CONCAT(DISTINCT fp.name ORDER BY fp.name SEPARATOR ', ') AS GT_POS
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN data dt
         ON se.fk_session_id = dt.fk_session_id
        LEFT JOIN reference_data_hist rd
         ON dt.data_id = rd.fk_data_id
        LEFT JOIN feature_param fp
         ON rd.fk_feature_param_id = fp.feature_param_id
        LEFT JOIN feature_param_type t
          ON fp.fk_feature_param_type_id = t.feature_param_type_id
        LEFT JOIN gt_recorder gt
          ON fp.feature_param_id = gt.fk_feature_param_id
      WHERE st.name != 'OCC' AND t.name = 'gt' AND rd.bin = 1 {1}
      GROUP BY session_id
      ) AS SESSION_GT_POS
      ON ss.session_id = SESSION_GT_POS.session_id


  LEFT JOIN
          (SELECT ss.session_id,
           GROUP_CONCAT(DISTINCT gt.opposite ORDER BY gt.opposite SEPARATOR ', ') AS GT_NEG
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN data dt
         ON se.fk_session_id = dt.fk_session_id
        LEFT JOIN reference_data_hist rd
         ON dt.data_id = rd.fk_data_id
        LEFT JOIN feature_param fp
         ON rd.fk_feature_param_id = fp.feature_param_id
        LEFT JOIN feature_param_type t
          ON fp.fk_feature_param_type_id = t.feature_param_type_id
        LEFT JOIN gt_recorder gt
          ON fp.feature_param_id = gt.fk_feature_param_id
      WHERE st.name != 'OCC' AND t.name = 'gt' AND rd.bin = 0 {1}
      GROUP BY session_id
      ) AS SESSION_GT_NEG
      ON ss.session_id = SESSION_GT_NEG.session_id



  LEFT JOIN
    ( SELECT ss.session_id,
                          GROUP_CONCAT(DISTINCT f.name ORDER BY f.name SEPARATOR ', ') as State
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN join_session_feature_param jsf
          ON ss.session_id = jsf.fk_session_id
        LEFT JOIN feature_param f
          ON jsf.fk_feature_param_id = f.feature_param_id
        LEFT JOIN feature_param_type t
          ON f.fk_feature_param_type_id = t.feature_param_type_id
      WHERE st.name != 'OCC' AND t.name = 'state' AND jsf.value = 'TRUE' {1}
      GROUP BY session_id
    ) AS SESSION_STATE
      ON ss.session_id = SESSION_STATE.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(sw.name ORDER BY se.setup_id SEPARATOR ', ') AS Version
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN test te
         ON se.setup_id = te.fk_setup_id
        LEFT JOIN sw_version sw
         ON te.fk_sw_version_id = sw.sw_version_id
      WHERE st.name != 'OCC' AND ((te.res_fpath LIKE '%usb_results%' AND te.res_fpath NOT LIKE '%NES_results%') OR
      (te.res_fpath NOT LIKE '%usb_results%' AND te.res_fpath LIKE '%NES_results%') OR
      (te.res_fpath LIKE '%mqtt_results%'
      AND te.res_fpath NOT LIKE '%usb_results%' AND te.res_fpath NOT LIKE '%NES_results%')) {1}
      GROUP BY session_id
    ) AS SESSION_VERSION
      ON ss.session_id = SESSION_VERSION.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(mu.name ORDER BY se.setup_id SEPARATOR ', ') AS Location
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_mount_area mu
         ON se.fk_nes_mount_area_id = mu.nes_mount_area_id
      WHERE st.name != 'OCC' {1}
      GROUP BY session_id
    ) AS SETUP_LOCATION
      ON ss.session_id = SETUP_LOCATION.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(nsp.name ORDER BY se.setup_id SEPARATOR ', ') AS Target
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_subject_position nsp
         ON se.fk_nes_subject_position_id = nsp.nes_subject_position_id
      WHERE st.name != 'OCC' {1}
      GROUP BY session_id
    ) AS SETUP_TARGET
      ON ss.session_id = SETUP_TARGET.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(jnc.value ORDER BY se.setup_id SEPARATOR ', ') AS Mode
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_config nc
         ON se.fk_nes_config_id = nc.nes_config_id
        LEFT JOIN join_nes_config_param jnc
         ON nc.nes_config_id = jnc.fk_nes_config_id
        LEFT JOIN nes_param np
         ON jnc.fk_nes_param_id = np.nes_param_id
      WHERE st.name != 'OCC' AND np.name = 'systemConfig_mode' {1}
      GROUP BY session_id
    ) AS SETUP_MODE
      ON ss.session_id = SETUP_MODE.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(jnc.value ORDER BY se.setup_id SEPARATOR ', ') AS FW
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_config nc
         ON se.fk_nes_config_id = nc.nes_config_id
        LEFT JOIN join_nes_config_param jnc
         ON nc.nes_config_id = jnc.fk_nes_config_id
        LEFT JOIN nes_param fw
         ON jnc.fk_nes_param_id = fw.nes_param_id
      WHERE st.name != 'OCC' AND fw.name = 'infoFW_version' {1}
      GROUP BY session_id
    ) AS SETUP_FW
      ON ss.session_id = SETUP_FW.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(nm.name ORDER BY se.setup_id SEPARATOR ', ') AS Model
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_config nc
         ON se.fk_nes_config_id = nc.nes_config_id
        LEFT JOIN nes_model nm
         ON nc.fk_nes_model_id = nm.nes_model_id
      WHERE st.name != 'OCC' {1}
      GROUP BY session_id
    ) AS SETUP_MODEL
      ON ss.session_id = SETUP_MODEL.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(se.target_distance_mm ORDER BY se.setup_id SEPARATOR ', ') AS Distance
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
      WHERE st.name != 'OCC' {1}
      GROUP BY session_id
    ) AS SETUP_DISTANCE
      ON ss.session_id = SETUP_DISTANCE.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(DISTINCT ci.serial_num ORDER BY se.setup_id SEPARATOR ', ') AS SN
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN nes_config nc
         ON se.fk_nes_config_id = nc.nes_config_id
        LEFT JOIN join_nes_config_component_inventory jcc
         ON nc.nes_config_id = jcc.fk_nes_config_id
        LEFT JOIN component_inventory ci
         ON jcc.fk_component_inventory_id = ci.component_inventory_id
      WHERE st.name != 'OCC' {1}
      GROUP BY session_id
    ) AS SETUP_SN
      ON ss.session_id = SETUP_SN.session_id


  LEFT JOIN
    ( SELECT ss.session_id,
             GROUP_CONCAT(dv.name ORDER BY se.setup_id SEPARATOR ', ') AS Validity
      FROM session ss
        JOIN setup se
         ON ss.session_id = se.fk_session_id
        LEFT JOIN setup_type st
         ON se.fk_setup_type_id = st.setup_type_id
        LEFT JOIN data d
         ON se.setup_id = d.fk_setup_id
        LEFT JOIN data_validity dv
         ON d.fk_data_validity_id = dv.data_validity_id
        LEFT JOIN sensor sn
         ON d.fk_sensor_id = sn.sensor_id
      WHERE st.name != 'OCC' AND sn.name = 'NES' AND (LENGTH(d.fpath)-LENGTH(REPLACE(d.fpath,'.','')))/LENGTH('.') = 1 
      AND ((substring_index(d.fpath, '.', -1) = 'ttlog') OR (substring_index(d.fpath, '.', -1) = 'tlog') OR 
      (substring_index(d.fpath, '.', -1) = 'blog')) {1}
      GROUP BY session_id
    ) AS SETUP_NES_VALIDITY
      ON ss.session_id = SETUP_NES_VALIDITY.session_id""".format(ss, se, WINDOWS_DIR)
            self._execute(query)
            res = self.cursor.fetchall()
            return [{'session': row[0], 'setup': row[1], 'time': row[2], 'duration': row[3], 'subject': row[4],
                     'gender': row[5], 'scenario': row[6], 'path': row[7].replace("//", "\\"), 'location': row[8],
                     'target': row[9], 'posture': row[10], 'state': str(row[11]).replace('is_', ''), 'gt': row[12],
                     'operator': row[13], 'mode': row[14],
                     'fw': row[15], 'model': row[16], 'distance': row[17], 'sn': row[18], 'version': row[19],
                     'company': row[20], 'notes': row[21], 'validity': row[22]} for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_note(self, setup: int) -> Optional[str]:
        """ return session note by setup id
            :param int setup: Unique identifier for the setup
            :return: session's note
            :rtype: str
            """
        try:

            self._execute("""SELECT session.note FROM setup INNER JOIN session ON 
            session.session_uuid = setup.fk_session_uuid AND setup.setup_id = %s""", [setup])
            res = self.cursor.fetchall()
            return res[0][0]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return None

    def setup_by_scenario(self, scenario: str) -> List[int]:
        """ Retrieve setup ids for which scenario exist

            :param str scenario: scenario to search
            :return: List of setup ids
            :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM session, setup, scenario, join_session_scenario WHERE 
            setup.fk_session_uuid = session.session_uuid AND 
            session.session_uuid = join_session_scenario.fk_session_uuid AND 
            scenario.scenario_uuid = join_session_scenario.fk_scenario_uuid AND 
            scenario.name LIKE %s ORDER BY setup.setup_id""", ["%" + scenario + "%"])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def setup_by_subject(self, subject: str) -> List[int]:
        """ Retrieve setup ids for which subject exist

            :param str subject: subject name
            :return: List of setup ids
            :rtype: List[int]
        """
        try:

            self._execute("""SELECT setup.setup_id FROM session, setup, subject WHERE 
            setup.fk_session_uuid = session.session_uuid AND session.fk_subject_uuid = subject.subject_uuid AND 
            subject.name LIKE %s ORDER BY setup.setup_id""", ["%" + subject + "%"])
            res = self.cursor.fetchall()
            return [row[0] for row in res]
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return []

    def __del_ref_hist(self, data_uuid: str, param_uuid: str) -> Optional[bool]:
        """ delete data from 'reference_data_hist' table
        :param str data_uuid: data UUID
        :param str param_uuid: param UUID
        :return: False/True
        :rtype: bool
        """
        try:
            affected_count = self._execute("""DELETE FROM reference_data_hist WHERE fk_data_uuid = %s AND 
            fk_feature_param_uuid = %s""", [data_uuid, param_uuid])
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return False

    def __insert_ref_hist(self, data: list) -> Optional[bool]:
        """ insert data to 'reference_data_hist' table
        :param list data: data to insert
        :return: False/True
        :rtype: bool
        """
        try:
            q = """INSERT INTO `reference_data_hist`(`fk_data_uuid`, `fk_data_id`, `fk_feature_param_uuid`, 
            `fk_feature_param_id`, `bin`,`value`) VALUES (%s, %s, %s, %s, %s, %s)"""
            affected_count = self.cursor.executemany(q, data)
            # Commit your changes in the database
            self.conn.commit()
            if affected_count > 0:
                return True
            return False
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            return False

    def set_reference(self, setup: int, sensor: Sensor, file: str) -> bool:
        """ set reference file given setup id and sensor name
        :param int setup: Unique identifier for the setup
        :param Sensor sensor: sensor name
        :param str file: path to file
        :return: False/True
        :rtype: bool
        """

        def __hist(gt_data: list) -> dict:
            """ compute gt data histogram
            :param list gt_data: gt data
            :return: hist
            :rtype: dict
            """
            hist = {}
            for x in gt_data:
                if round(x) in hist:
                    hist[round(x)] += 1
                else:
                    hist[round(x)] = 1
            return hist

        def __build_gt(gt_file: str) -> dict:
            """ build gt data from gt recorder file
            :param list gt_file: gt recorder file
            :return: gt dict data <gt: data>
            :rtype: dict
            """

            def __label(res: list, gt_lines: list, gt_param: tuple):
                """ label gt data given gt lines from gt recorder file and gt param
                :param list gt_lines: gt lines from gt recorder file
                :param tuple gt_param: gt param tuple
                :param list res: res
                :rtype: void
                """
                last_sec = 0
                last_label = None
                for line in gt_lines:
                    if gt_param[0] in line:
                        search = gt_param[0]
                        label = 0.0
                    elif gt_param[1] in line:
                        search = gt_param[1]
                        label = 1.0
                    else:
                        search = "session ends"
                        label = None
                    sch = re.search(search + '(.*): ', line).group(0)
                    sec = int(line.split(sch)[1].strip('\n'))
                    res.extend([last_label] * (sec - last_sec))
                    last_sec = sec
                    last_label = label

            with open(gt_file) as f:
                lines = f.readlines()
            rest = []
            occupancy = []
            stationary = []
            speaking = []
            zrr = []
            rest_lines = [line for line in lines if ('rest' in line or 'motion' in line)] + [lines[-1]]
            occ_lines = [line for line in lines if ('full' in line or 'empty' in line)] + [lines[-1]]
            stat_lines = [line for line in lines if ('stationary' in line or 'driving' in line)] + [lines[-1]]
            speak_lines = [line for line in lines if ('not speaking' in line or 'speaking' in line)] + [lines[-1]]
            zrr_lines = [line for line in lines if ('zero rr' in line or 'respiration' in line)] + [lines[-1]]

            __label(rest, rest_lines, ('motion', 'rest'))
            __label(occupancy, occ_lines, ('empty', 'full'))
            __label(stationary, stat_lines, ('driving', 'stationary'))
            __label(speaking, speak_lines, ('not speaking', 'speaking'))
            __label(zrr, zrr_lines, ('respiration', 'zero rr'))

            return {'rest': rest, 'occupancy': occupancy, 'stationary': stationary, 'speaking': speaking, 'zrr': zrr}

        try:
            file_name = os.path.basename(file)
            if sensor == Sensor.gt_recorder:
                pth_list = self.setup_ref_path(setup=setup, sensor=sensor)
                pth = pth_list[0] if pth_list else None
                assert file_name == os.path.basename(pth), "File name should be equal to legacy file in DB!"
                assert os.path.exists(file), "File not exists!"
                setup_uuid = self._setup_uuid_by_id(setup)
                session_uuid = self.__session_uuid(setup)
                if setup_uuid and session_uuid and pth:
                    dict_gt = __build_gt(file)
                    for gt in dict_gt:
                        sens = 'gt_' + gt
                        sensor_uuid = self.__sensor_ids(Sensor[sens])[0]
                        if sensor_uuid:
                            list_data_ids = self.__data_ids(session_uuid, sensor_uuid)
                            param_uuid, param_id = self._feature_param_ids(gt)
                            if list_data_ids and param_uuid:
                                data_ids = list_data_ids[0]
                                hist_ref = __hist(dict_gt[gt])
                                data = [[data_ids[0], data_ids[1], param_uuid, param_id, key, hist_ref[key]] for key
                                        in hist_ref]
                                val_del = self.__del_ref_hist(data_ids[0], param_uuid)
                                if not val_del:
                                    return False
                                val_insert = self.__insert_ref_hist(data)
                                if not val_insert:
                                    return False
                                ref_list = self.setup_ref_path(setup=setup, sensor=Sensor[sens])
                                pth_ref = ref_list[0] if ref_list else None
                                if not pth_ref:
                                    return False
                                if file != pth:
                                    copyfile(file, pth)
                                np.save(pth_ref, dict_gt[gt])  # overwrite npy array
                    return True
        except IOError as err:
            print(err)
        except AssertionError as err:
            print(err)
        return False

    def __del__(self):
        """
        Closes connection to MySQL DB
        """
        self.conn.close()


if __name__ == '__main__':


    db = DB('neteera_db')   # old db

    db.update_mysql_db(9816)    # change to the db that has the setup 9816

    db.setup_ref_path(9838, 'natus')

    print(db.setup_ts(9838, Sensor.nes))
    # print(db.setup_ref_path_npy(setup=3307, sensor=Sensor.nes_res, vs=VS.hr))  # get npy pth by setup + sensor + vs
    exit()
    db.update_mysql_db(4815)    # change to the db that has the setup 4815
    print(db.setup_duration(setup=4815))  # get duration by setup id
    # # run examples:
    print(db.is_VSM(setup=629))  # check if setup is of type 'vsm'
    print(db.setup_env(setup=6231))  # get environment by setup id
    print(db.setup_seat(1675))  # get seat type by setup id
    print(db.setup_ref(setup=629))  # get a list of setup reference sensors
    print(db.setup_ext_proc_ref(setup=4047))  # get a list of setup reference sensors excluding 'ADDITIONAL_SENSORS'
    print(db.setup_target(setup=2))  # get target by setup id
    print(db.setup_mount(setup=2))  # get mount by setup id
    print(
        db.setup_ts(setup=6469, sensor=Sensor.nes))  # get ts by setup id and sensor name (dict containing start/end ts)
    print(db.setup_ts(setup=5852, sensor=Sensor.epm_10m, search='ECG'))  # optional search for specifying data file ts
    print(db.setup_fs(setup=6281))  # get sampling freq by setup id
    print(db.setup_fw(setup=5159))  # get firmware by setup id
    print(db.setup_subject(setup=43))  # get subject name by setup id
    print(db.setup_operator(setup=5852))  # get operator name by setup id
    print(db.setup_posture(setup=4318))  # get subject posture by setup id
    print(db.setup_subject_details(2427))  # get subject gender, birth year, height, weight by setup id (dict)
    print(db.setup_mode(setup=3171))  # get radar configuration mode by setup
    print(db.setup_distance(setup=3171))  # get radar distance from target(mm) by setup id
    print(db.setup_bin(setup=5966))  # get bin params {range, bin} by setup id
    print(db.setup_spot(setup=5966))  # get spot params {type, t0 (start time in sec), X (duration in sec)}by setup id
    print(db.setup_spot_res(setup=3252))  # get spot values by setup id as dict
    print(db.setup_radar_config(setup=3423))  # get radar config key value by setup id
    print(db.setup_ADC_clkDiv(setup=3460))  # get 'ADC_clkDiv' param from raw nes data file
    print(db.setup_dir(setup=4390))  # get setup base_dir_folder by setup id
    print(db.setup_delay(setup=5852, sensor=Sensor.epm_10m))  # get delay between nes and reference (in sec)
    print(db.setup_delay(setup=5852, sensor=Sensor.epm_10m,
                         search='ECG'))  # optional search for specifying data file delay
    print(db.setup_sw_version(setup=4107))  # get sw version by setup id
    print(db.setup_sn(setup=4106))  # get sn by setup id
    print(db.setup_note(setup=4662))  # # get note by setup id
    print(db.setup_ref_path(setup=335,
                            sensor=Sensor.epm_10m))  # get data fpath by (db logger + legacy)setup_id and sensor_name
    print(db.setup_ref_path(setup=11, sensor=Sensor.nes))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path(setup=11, sensor=Sensor.gt_occupancy))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path(setup=11, sensor=Sensor.gt_recorder))  # get data fpath by setup_id and sensor_name
    # get data fpath by setup + sensor + file type
    print(db.setup_ref_path(setup=3457, sensor=Sensor.nes, extension=FileType.cfg))
    print(db.setup_ref_path(setup=3846, sensor=Sensor.epm_10m, extension=FileType.csv, search='ECG'))  # get data fpath
    # by setup + sensor + file type + search file name str

    print(db.setup_view())  # get setup fields according to the db view page
    print(db.setup_view(setup=3846))  # get setup fields according to the db view page by setup id

    # get data fpath by setup + sensor
    print(db.setup_ref_path(setup=576, sensor=Sensor.biopac))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path(setup=1726, sensor=Sensor.zephyr_bw))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path(setup=1726, sensor=Sensor.zephyr_ecg))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path(setup=1726, sensor=Sensor.zephyr_hr_rr))  # get data fpath by setup_id and sensor_name
    print(db.setup_ref_path_npy(setup=3307, sensor=Sensor.nes, vs=VS.raw))  # get npy pth by setup + sensor + vs
    print(db.setup_ref_path_npy(setup=3307, sensor=Sensor.biopac, vs=VS.hr))  # get npy pth by setup + sensor + vs
    print(db.setup_ref_path_npy(setup=3307, sensor=Sensor.nes_res, vs=VS.hr))  # get npy pth by setup + sensor + vs
    print(db.setup_data_validity(setup=694, sensor=Sensor.nes))  # get data validity for setup-sensor
    print(db.setup_data_validity(setup=527, sensor=Sensor.biopac, vs=VS.rr))  # get data validity for setup-sensor-vs

    print(db.setup_from_to_date(from_date='2020-08-30 00:00:00', to_date='2020-08-30 23:59:59'))  # get a list of setup
    # ids by date. Format '%Y-%m-%d' hh::mm:ss
    print(db.agc_setups())  # retrieve setup ids for which AGC control is enabled
    print(db.setup_multi(setup=3621))  # get a list of setup ids for a multi session by setup id

    print(db.setup_by_project(prj=Project.elbit))  # get list of setup ids by project name
    print(db.setup_project_id(setup=1892, prj=Project.elbit))  # get project setup id by setup id and project name
    print(db.setup_by_project_id(prj=Project.elbit, pid=94))  # get setup id by project name and project setup id
    print(db.setup_by_epoch(epoch='1620723454713'))  # get setup id by epoch ts

    print(db.setup_by_vs(vs=VS.bp))  # get a list of setup ids by vs type
    print(db.setup_by_vs(vs=VS.hri, sensor=Sensor.epm_10m))  # get a list of setup ids by vs + sensor type
    # get a list of setup ids by reference data
    print(db.setup_by_sensor(sensor=Sensor.epm_10m))
    print(db.setup_by_sensor(sensor=Sensor.gt_zrr))
    print(db.setup_by_sensor(sensor=Sensor.spo2))
    print(db.setup_by_sensor(sensor=Sensor.biopac))
    print(db.setup_by_sensor(sensor=Sensor.gt_occupancy))

    # get a list of setup ids by scenario
    print(db.setup_by_scenario('Mild motion'))

    # get a list of setup ids by subject
    print(db.setup_by_subject('Moshe Aboud'))

    # get a list of setup ids for which reference clk sync exists
    print(db.setup_by_clk(sensor=Sensor.biopac))
    print(db.setup_by_clk(sensor=Sensor.gt_rest))

    # get a list of setup ids for which nes-reference delay exists
    print(db.setup_by_delay(sensor=Sensor.biopac))
    print(db.setup_by_delay(sensor=Sensor.gt_rest))

    # get setup ids by firmware version
    print(db.setup_fw_equals(fw='0.1.3.1'))
    print(db.setup_fw_smaller_than(fw='0.3.9.3'))
    print(db.setup_fw_greater_than(fw='0.3.9.3'))

    # get a list of setup ids by number of bins for cpx data only
    print(db.setup_bins_smaller_than(value=5))
    print(db.setup_bins_greater_than(value=5))
    print(db.setup_bins_equals(value=5))
    print(db.setup_bins_in_range(minn=1, maxx=5))
    # get a list of setup ids by bin offset for cpx data only
    print(db.setup_offset_smaller_than(value=5))
    print(db.setup_offset_greater_than(value=1))
    print(db.setup_offset_equals(value=1))
    print(db.setup_offset_in_range(minn=1, maxx=5))
    #  combination example
    bins = set(db.setup_bins_equals(value=5))
    offset = set(db.setup_offset_equals(value=1))
    print(sorted(list(bins & offset)))

    # get a list of setup ids by distance in mm
    print(db.setup_distance_smaller_than(distance=100))
    print(db.setup_distance_greater_than(distance=100))
    print(db.setup_distance_equals(distance=100))
    print(db.setup_distance_in_range(minn=100, maxx=1000))

    print(db.benchmark_setups(benchmark=Benchmark.covid_belinson_sample,
                              value=True))  # get a list of setup ids for covid_belinson_trials
    print(db.benchmark_setups(benchmark=Benchmark.fae_rest))

    print(db.ir_setups())  # get a list of interferometer setups
    print(db.sr_setups())  # get a list of radar setups
    print(db.sr_cw_setups())  # get a list of cw radar setups
    print(db.sr_fmcw_setups())  # get a list of fmcw radar setups
    print(db.all_setups())  # get a list of setup ids

    # get a list of setup ids by sensor/vs validity
    print(db.setup_by_data_validity(sensor=Sensor.gt_rest, value=Validation.confirmed))
    print(db.setup_by_data_validity(sensor=Sensor.gt_rest, value=Validation.confirmed))
    print(db.setup_by_data_validity(vs=VS.rr, value=Validation.valid))
    print(db.setup_by_data_validity(sensor=Sensor.biopac, vs=VS.rr, value=Validation.valid))
    print(db.setup_by_data_validity(value=Validation.confirmed))

    print(db.setup_by_version(version='1.9.3.1'))  # get a list of setup ids by version
    print(db.setup_by_note(note='Cherry'))  # get a list of setup ids by note
    print(db.setup_by_model(model=Model.cherry))  # get a list of setup ids by model

    # sensor-vs correlation
    print(db.vs_by_sensor(sensor=Sensor.biopac))  # get vs by sensor
    print(db.sensor_by_vs(vs=VS.hr))  # get sensor by vs
    print(db.sensor_by_vs(vs=VS.zrr))  # get sensor by vs

    # insert to DB
    # print(db.set_benchmark(setup=200, benchmark=Benchmark.for_test, value=True))  # set benchmark 'for_test'
    # print(db.set_benchmark(setup=200, benchmark=Benchmark.jenkins, value=True))  # set benchmark 'jenkins'
    # print(db.set_benchmark(setup=200, benchmark=Benchmark.for_test, value=False))  # unset benchmark 'for_test'
    # print(db.set_delay(setup=360, sensor=Sensor.biopac, value=-4.5))  # set delay between nes and reference (in sec)
    # optional search for specifying data file delay
    # print(db.set_delay(setup=5852, sensor=Sensor.epm_10m, value=1, search='ECG'))
    # print(db.set_data_validity(setup=1691, sensor=Sensor.biopac, value=Validation.invalid))  # set data validity for a
    # given sensor
    # print(db.set_data_validity(setup=1691, sensor=Sensor.biopac, vs=VS.rr, value=Validation.invalid))  # set data
    # validity for a given sensor & vs
    # print(db.set_state(setup=1732, state=State.is_motion, value=False))  # set state value for a given setup id and
    # state name
    # print(db.set_note(setup=1899, note='DMS disabled'))  # set session note for a given setup id
    # print(db.set_setup_project_id(setup=1969, prj=Project.elbit, value='113'))  # set project setup id
    # print(db.set_subject_gender(setup=6080, gender=Gender.female))  # set subject gender given setup id
    # print(db.set_subject_age(setup=6080, age=1)) # set subject age given setup id
    # print(db.set_subject_height(setup=6080, height=161)) # set subject height given setup id
    # print(db.set_subject_weight(setup=6080, weight=81)) # set subject weight given setup id
    # print(db.set_fs(setup=3270, fs=2500.0))  # set fs for a given setup id
    # print(db.set_setup_subject(setup=3941, subject='Ohad Basha'))  # set subject for a given setup id, subject name
    # print(db.set_setup_target(setup=3941, target=Target.back))  # set target for a given setup id, target
    # print(db.set_setup_mount(setup=3941, mount=Mount.seat_back))  # set mount for a given setup id, mount
    # print(db.set_setup_sn(setup=4371, sn='131020450019'))  # set sn for a given setup id, s/n ('Myrtus' and above)
    # print(db.insert_setup_scenario(setup=4361, scenario="Zero RR scenario"))  # add scenario for a given setup id,
    # scenario name
    # print(db.insert_sn(132021440507, 'MyrtusA1', 'Neteera', 'my note'))
    # print(db.insert_subject('Cool Guy', 'Human', 'Neteera', 1970, 180, 80, 'Male', 'cool note'))
    # print(db.remove_setup_scenario(setup=4361, scenario="Zero RR scenario"))  # remove scenario for a given setup id,
    # scenario name
    # print(db.set_posture(setup=4435, posture=Posture.standing))  # set subject posture for a given setup id
    # print(db.set_distance(setup=4435, distance=1000))  # set setup distance for a given setup id
    # print(db.set_model(setup=4435, model='SiR_L6'))  # set setup model for a given setup id
    # print(db.set_company(setup=4435, prj=Project.neteera))  # set setup company for a given setup id
    # print(db.set_reference(setup=8078, sensor=Sensor.gt_recorder, file=r""))  # set reference file


    # get a list of setup ids by state
    print(db.setup_by_state(state=State.is_rest, value=True))
    print(db.setup_by_state(state=State.is_empty, value=True))
    print(db.setup_by_state(state=State.is_motion, value=True))
    print(db.setup_by_state(state=State.is_hb, value=True))
    print(db.setup_by_state(state=State.is_occupied, value=True))
    print(db.setup_by_state(state=State.is_stationary, value=False))
    print(db.setup_by_state(state=State.is_engine_on, value=False))
    print(db.setup_by_state(state=State.is_driving, value=False))
    print(db.setup_by_state(state=State.is_driving_idle, value=False))
    print(db.setup_by_state(state=State.is_speaking, value=True))
    print(db.setup_by_state(state=State.is_occupancy, value=False))  # for occupancy presence

    # get a list of setup ids by nes target (subject position in relation to radar)
    print(db.setup_by_target(target=Target.back))
    print(db.setup_by_target(target=Target.bottom))
    print(db.setup_by_target(target=Target.front))
    print(db.setup_by_target(target=Target.top))
    print(db.setup_by_target(target=Target.side))

    # get a list of setup ids by mount
    print(db.setup_by_mount(mount=Mount.seat_back))
    print(db.setup_by_mount(mount=Mount.seat_bottom))
    print(db.setup_by_mount(mount=Mount.seat_side))
    print(db.setup_by_mount(mount=Mount.seat_headrest))
    print(db.setup_by_mount(mount=Mount.bed_bottom))
    print(db.setup_by_mount(mount=Mount.bed_side))
    print(db.setup_by_mount(mount=Mount.lab_ceiling))
    print(db.setup_by_mount(mount=Mount.vehicle_ceiling))
    print(db.setup_by_mount(mount=Mount.lab_other))

    # get a list of setup ids by posture
    print(db.setup_by_posture(posture=Posture.sitting))

    # get a list of setup ids by environment
    print(db.setup_by_environment(environment=Environment.lab))

    # get vs avg for a given setup id, sensor name and vs name
    print(db.setup_vs_avg(setup=1771, sensor=Sensor.biopac, vs=VS.hr))

    # get setups by histogram vs values
    print(db.setup_vs_smaller_than(vs=VS.hr, value=60))
    print(db.setup_vs_greater_than(vs=VS.hr, value=70))
    print(db.setup_vs_equals(vs=VS.hr, value=100))
    print(db.setup_vs_in_range(vs=VS.hr, minn=60, maxx=80))

    # example of a combination
    # get setups with 'hr' grater than 100 and between 90-95
    values_1 = set(db.setup_vs_greater_than(vs=VS.hr, value=100))
    values_2 = set(db.setup_vs_in_range(vs=VS.hr, minn=90, maxx=95))
    inter = values_1 & values_2  # '&' operator is used for set intersection

    # get setups with 'hr' grater than 100 or between 90-95
    values_1 = set(db.setup_vs_greater_than(vs=VS.hr, value=100))
    values_2 = set(db.setup_vs_in_range(vs=VS.hr, minn=90, maxx=95))
    union = values_1 | values_2  # '|' operator is used for set unification
