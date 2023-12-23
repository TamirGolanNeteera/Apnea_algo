# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
from packaging import version

from Tests.Constants import DELIVERED
from Tests.Utils.PathUtils import change_dir_to_pythondev, create_dir
from Tests.Utils.StringUtils import join
from Tests.vsms_db_api import *

import sys
from typing import Tuple, Iterable
import os
import platform
import subprocess
import numpy as np
import pandas as pd

PYTHONVERSION = 'python3'  # '/usr/bin/python3.7'


def shift_reliability(reliability, gt_len, shift):
    if len(reliability):
        if shift < 0:
            reliability = reliability[abs(shift):abs(shift) + gt_len]
        else:
            reliability = reliability[:gt_len]
    return reliability


def first_ind(prediction):
    if prediction is None:
        return 
    if np.max(prediction) <= 0:
        return len(prediction) + 1
    return [x[0] for x in enumerate(prediction) if x[1] > 0][0]


def remove_tails(*vecs) -> (iter, iter):
    min_len = min(len(x) for x in vecs)
    return [vec[:min_len] for vec in vecs]


def match_lists(prediction: pd.DataFrame, g_truth: list, max_shift: int = 30) -> Tuple[iter, list, int]:
    """ Synchronize two lists

    :param list prediction: A list of numbers
    :param list g_truth: A list of numbers
    :param int max_shift: Maximum to shift `prediction` so as to synchronize it with `g_truth` (in seconds)
    """

    def mean_absolute_errors(a, b):
        max_shift_to_test = min(max_shift, len(a), len(b))
        return [np.nanmean(np.abs(a[i:i + len(b)] - b[:len(a) - i])) for i in range(max_shift_to_test)]

    if not list(prediction) or not list(g_truth) or len(prediction) == 1 or len(g_truth) == 1:  # spot mode
        return prediction, g_truth, 0

    # todo call match_lists_by_ts from here
    pred_using_reliability = prediction
    if 'high_qual' in prediction and np.sum(prediction['high_qual']) > 15:
        pred_using_reliability =\
            [pred if prediction['high_qual'][i] else np.nan for i, pred in enumerate(prediction)]
    pred_using_reliability = np.array(pred_using_reliability)
    g_truth = np.array(g_truth)
    mse_shift_pred = mean_absolute_errors(pred_using_reliability, g_truth)
    mse_shift_gt = mean_absolute_errors(g_truth, pred_using_reliability)
    if np.nanmin(mse_shift_gt) < np.nanmin(mse_shift_pred):
        shift = mse_shift_gt.index(np.nanmin(mse_shift_gt))
        g_truth = g_truth[shift:]
        final_shift = shift
    else:
        try:
            shift = mse_shift_pred.index(np.nanmin(mse_shift_pred))
        except ValueError:
            return [], [], 0,
        prediction = prediction[shift:]
        final_shift = -shift
    min_len = min(len(g_truth), len(prediction))
    g_truth = g_truth[:min_len]
    prediction = prediction[:min_len]
    return prediction, g_truth, final_shift


def run_cmd(cmd):
    try:
        change_dir_to_pythondev()
    except IndexError:  # todo check why
        pass
    python_ver = 'python3 ' if platform.system() == 'Linux' else 'python '
    cmd = python_ver + cmd.replace('python3 ', '').replace('python ', '')
    print('\n', cmd)
    subprocess.run(cmd, shell=True)


def create_version_name(args, benchmark_name: str) -> str:
    testing_version = f'{args.version}_{benchmark_name}'
    return testing_version


def intersect(list_of_lists: Iterable) -> list:
    """ Intersect lists
        WARNING: THIS FUNCTION CAN BE RANDOM!
    """
    if not list_of_lists:
        return []
    return sorted(list(set.intersection(*map(set, list_of_lists))))


def collect_result(setups, destination):
    create_dir(destination)
    if setups and len(os.listdir(destination)) == 0:
        change_dir_to_pythondev()
        run_cmd(f' ./Tests/CPPTools/CollectResults.py -setups {join(setups)} -save_path {destination}')


def get_setups_list(folder, vs):
    return [re.findall(r'\d+', file)[0] for file in os.listdir(folder) if re.fullmatch(fr"{vs}_[0-9]+\.png", file)]


def is_debug_mode():
    get_trace = getattr(sys, 'gettrace', None)
    return get_trace() is not None


def note_by_sn_for_standing(idw, db):
    sn_last_digits = db.setup_sn(idw)[0][-2:]
    note = db.setup_note(idw).replace('\t', '').replace('\n', ', ')
    if 'SN' in note and sn_last_digits in note:
        first_sn = note.find('SN')
        last_sn = note.rfind('SN')
        sn_index = note.find(sn_last_digits)
        if sn_index < last_sn:
            note = note[:last_sn - 1]
            if note.endswith(','):
                note = note[:-1]
        else:
            note = note[:first_sn] + note[last_sn:]
    return note


def last_delivered_version():
    """get the last(not latest) algo version"""
    newest_number = max([version.parse(x[8:]) for x in os.listdir(DELIVERED) if x.startswith('net-alg')])
    return 'net-alg-' + str(newest_number)
