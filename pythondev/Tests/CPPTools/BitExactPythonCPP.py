# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
# http://wiki.neteera.local/index.php?title=BitExactPythonCPP
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa

from Tests.Utils.ResearchUtils import print_var
from Tests.CPPTools.BitExactEvalutor import get_setup_list
from Tests.Utils.LoadingAPI import load_pred_rel_from_npy, load_pred_rel_from_csv

import numpy as np
import pandas as pd
import argparse
import os

EPSILON = 0.01


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cpp', type=str, help='cpp folder results', required=True)
    parser.add_argument('-py', type=str, help='cpp folder results')
    parser.add_argument('-epsilon', type=float, default=EPSILON)

    return parser.parse_args()


def bit_exact_columns_float(col1, col2, epsilon, axis=None):
    return np.all(np.abs(col1 - col2) < epsilon, axis=axis)


def bit_exact_columns_int_str(col1, col2, epsilon, axis=None):
    return np.all(col1 == col2, axis=axis)


def compare_results(bit_exact_function, cpp_result, py_result, vs, setup, epsilon=EPSILON):
    if len(cpp_result) != len(py_result):
        print(f'vs {vs}, setup {setup}. Results are not of the same length')
        print_var(len(cpp_result))
        print_var(len(py_result))
        return False
    else:
        if not bit_exact_function(cpp_result, py_result, epsilon):
            print(f'vs {vs}, setup {setup}. Results are not bit exact!!!')
            print_df = pd.concat((
                py_result.rename(columns=lambda x: f'{x}_py'), cpp_result.rename(columns=lambda x: f'{x}_cpp')), axis=1)
            print_df['bit_exact'] = bit_exact_function(py_result, cpp_result, epsilon=EPSILON, axis=1)
            print(print_df)
            return False
    return True


def check_dynamic_mode(vs, cpp_path, py_path):
    print(f'\nchecking dynamic mode vs {vs}')
    bit_exact_count = 0
    total = 0
    cpp = os.path.join(cpp_path, 'dynamic')
    py = os.path.join(py_path, 'dynamic')
    if vs in ['hr', 'rr', 'stat']:
        bit_exact_function = bit_exact_columns_int_str
    else:
        bit_exact_function = bit_exact_columns_float
    for setup in get_setup_list(cpp):
        cpp_result = load_pred_rel_from_csv(cpp, setup, vs)
        try:
            py_result = load_pred_rel_from_npy(py, setup, vs)
        except FileNotFoundError:
            continue
        if vs in ['hr', 'rr']:
            cpp_result = pd.DataFrame(cpp_result)
            py_result = pd.DataFrame(py_result)
        else:
            cpp_result = pd.DataFrame(cpp_result['pred'])
            py_result = pd.DataFrame(py_result['pred'])
        bit_exact = compare_results(bit_exact_function, cpp_result, py_result, vs, setup)
        bit_exact_count += int(bit_exact)
        total += 1
    print(f'{bit_exact_count} results were checked and found bit exact out of {total}')


def check_bit_exact(cpp_path, py_path, epsilon=EPSILON):
    pd.set_option("display.max_rows", 999)
    pd.set_option("display.width", 500)
    pd.set_option("display.max_columns", 20)
    for vs in ['hr', 'rr', 'ra', 'stat', 'ie']:
        check_dynamic_mode(vs, cpp_path, py_path)

    for vs in ['hr', 'rr']:
        print(f'\nchecking spot mode vs {vs}')
        setup_counter = 0
        cpp = os.path.join(cpp_path, 'spot')
        py = os.path.join(py_path, 'spot')
        for setup in get_setup_list(cpp):
            cpp_result = load_pred_rel_from_csv(cpp, setup, vs)['pred']
            try:
                py_result = load_pred_rel_from_npy(py, setup, vs)['pred']
            except FileNotFoundError:
                continue
                print(f'vs {vs}, setup {setup} not in py dir (if new or invalid then ignore)')
            bit_exact = compare_results(bit_exact_columns_int_str, cpp_result, py_result, vs, setup)
            setup_counter += int(bit_exact)
        if setup_counter:
            print(f'{setup_counter} results were checked and found bit exact')


if __name__ == '__main__':
    args = get_args()
    check_bit_exact(args.cpp, args.py, args.epsilon)
