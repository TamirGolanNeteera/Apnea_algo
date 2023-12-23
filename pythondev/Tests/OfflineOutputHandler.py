# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import logging

from Tests.Evaluation.VersionEvaluator import STAT_CLASSES, UNITS
from Tests.vsms_db_api import DB
from pylibneteera.datatypes import VitalSignType

from Tests.Utils.LoadingAPI import load_shifted_reference
from Tests.Accumulator import accumulate_return_values

import numpy as np


class OfflineOutputHandler:
    """
    Compares algorithm prediction with reference system

    :param int idx: setup number
    """

    def __init__(self, idx: int, db: DB, vital_signs):
        self._idx = idx
        self.last_time_printed = -1
        self._ref_data = {}
        for vs in vital_signs:
            try:
                self._ref_data[vs] = load_shifted_reference(idx, str(vs), db)
            except (AssertionError, IndexError, TypeError):  # no reference
                pass

    def print_time(self, sec: int, end_time: int):
        """ Prints Vital Sign Tracker result and Reference System result

        :param int sec: running index
        :param int end_time: end index
        """
        logging.getLogger('vsms').info(f'setup: {self._idx}   Second: {sec:.2f} / {end_time}  ')

    @accumulate_return_values
    def _acc_ref(self, x):
        return x

    def print_result(self, sec: int, vs_val, vs_type: VitalSignType, end_time=None) -> None:
        """ Prints Vital Sign Tracker result vs. Reference System result

        :param end_time: end time of the run (given as input argument or as the setups duration)
        :param int sec: running index
        :param vs_val: vital sign data and reference data
        :argument VitalSignType vs_type: type of the vital sign result
        """

        units = UNITS.get(vs_type.value, '')
        ref_data = self._ref_data.get(vs_type)
        reference_str = ''
        if ref_data is not None and len(ref_data) > sec:
            ref_value = ref_data[int(sec)]
            if ref_value is not None:
                if vs_type == VitalSignType.stat:
                    ref_value = STAT_CLASSES[int(ref_value)]
                else:
                    ref_value = np.round(ref_value, 2)
                reference_str = f';  Reference: {ref_value:>15} {units}'

        if sec > self.last_time_printed and end_time is not None:
            self.last_time_printed = sec
            self.print_time(sec, end_time=end_time)

        if vs_type in [VitalSignType.intra_breath]:
            logging.getLogger('vsms').info(f'{vs_type:<15}\t{vs_val} {reference_str}')
            return

        if vs_type in [VitalSignType.identity]:
            name = vs_val['identity']
            logging.getLogger('vsms').info(f'{vs_type:<15}\t{name} {reference_str}')
            return

        if vs_type in [VitalSignType.posture]:
            logging.getLogger('vsms').info(f'{vs_type:<15}\t{vs_val} {reference_str}')
            return

        reliability_str = ''
        if isinstance(vs_val, tuple) or isinstance(vs_val, list):
            if vs_val[1] in [True, False]:
                reliability_str = f'  Reliability: {vs_val[1]}   High qual: {vs_val[2]}'
            vs_val = vs_val[0]
        if isinstance(vs_val, float):
            vs_val = round(vs_val, 2)
        logging.getLogger('vsms').info(f'{vs_type:<7}\t{vs_val:>15} {units}  {reliability_str} {reference_str}')
