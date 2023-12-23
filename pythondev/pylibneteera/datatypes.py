# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from enum import Enum


class NamedEnum(Enum):
    def __str__(self) -> str:
        return self.name


class Status(NamedEnum):
    """ Statuses"""
    empty = 'empty'
    warming_up_after_empty = 'warming_up_after_empty'
    full = 'full'
    unknown = 'unknown'
    no_signal = 'no signal'
    zero_respiration = 'zero respiration'
    motion = 'motion'
    speaking = 'speaking'


class SensorType(NamedEnum):
    """ Supported sensor types"""
    bcg = 'bcg'
    biopac = 'biopac'
    gps = 'gps'
    spO2 = 'spO2'
    ecg = 'ecg'
    occupancy = 'occupancy'  # used by SessionList
    zrr = 'zrr'  # used by SessionList
    stat = 'stat'
    bitalino = 'bitalino'
    natus = 'natus'
    dn_ecg = 'dn_ecg'
    hexoskin = 'hexoskin'
    zephyr_ecg = 'zephyr_ecg'
    zephyr_bw = 'zephyr_bw'
    zephyr_hr_rr = 'zephyr_hr_rr'
    elbit_ecg_sample = 'elbit_ecg_sample'
    imotion_biopac = 'imotion_biopac'
    epm_10m = 'epm_10m'
    capnograph_pc900b = 'capnograph_pc900b'
    posture = 'posture'
    spirometer = 'spirometer'


class VitalSignType(NamedEnum):
    """ Supported vital signs, The order is the order of computation!"""
    rr = 'rr'
    hr = 'hr'
    ie = 'ie'
    ra = 'ra'
    stat = 'stat'
    intra_breath = 'intra_breath'
    bbi = 'bbi'  # beat-to-beat interval used in spot mode
    identity = 'identity'  # Bio-ID
    bp = 'bp'  # blood pressure, not supported
    posture = 'posture'
    sleep_stages = 'sleep_stages'
    apnea = 'apnea'


class NeteeraSensorType(NamedEnum):
    """ Type of Neteera sensors """
    cw = 'CW'
    fmcw = 'FMCW'


class NeteeraTargetType(NamedEnum):
    """ Type of Neteera sensor Targets"""
    front = 'front'
    back = 'back'
    bottom = 'bottom'
    top = 'top'
    side = 'side'


class NeteeraMountType(NamedEnum):
    """ Type of Neteera sensor Mounts"""
    bed = 'bed'
    seat = 'seat'
