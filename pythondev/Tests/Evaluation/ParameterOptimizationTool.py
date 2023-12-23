# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa

from Tests.Utils.TestsUtils import create_version_name
from Tests.Utils.PathUtils import copy_code_and_change_working_dir
from Tests.Utils.StringUtils import join
import Tests.VersionTester
from Tests.VersionTester import main_version_tester, version_eval
from Tests.VSParser import post_parsing
from Tests.vsms_db_api import DB

import os
import pandas as pd
import re
from time import sleep


def get_args():
    parser = Tests.VersionTester.get_parser()
    parser.add_argument('-parameter_name', type=str, help='parameter name as in Configurations.py')
    parser.add_argument('-original_value', type=str, help='parameter value as in Configurations.py')
    parser.add_argument('-values', type=str, nargs='+', help='values to test')
    parser.add_argument('-sleep', type=int, default=0, help='time to sleep before start')
    parser.add_argument('-units', type=str, default='', help='units of parameter')
    return post_parsing(parser.parse_args())


def replace_config(param_str, old, new, folder):
    os.chdir(os.path.join(folder, 'pythondev'))
    print(f'working dir changed to {os.getcwd()}')
    config_path = os.path.join(os.getcwd(), 'Configurations.py')
    with open(config_path, "r+") as f:
        data = f.read()
        param_value_str = param_str + str(old)
        occurrences_counter = data.count(param_value_str)
        print(f'{param_value_str} was found {occurrences_counter} times in Configurations.py')
        assert occurrences_counter
        data = data.replace(param_value_str, param_str + str(new))
        f.seek(0)
        f.write(data)
        f.truncate()


def strip_non_alphanumeric(string: str):
    """Turn all characters that are not: {digits, letters, '.', '_' , '-'} to '_' to compile with argparser, folder
    naming etc"""
    return re.sub(r'[^a-zA-Z0-9\.\_]', '_', str(string))


def optimize_parameter(args, db):
    sleep(args.sleep)
    args.values.append(args.original_value)
    version_name = args.version
    args.result_dir = os.path.join(args.result_dir, version_name)
    parameter_rgx = "'" + args.parameter_name + "': "
    copy_code_and_change_working_dir(args.result_dir)

    current_value = args.original_value
    version_dirs = pd.DataFrame()

    for v in args.values:
        args.version = strip_non_alphanumeric(current_value).replace('.', 'p')
        if args.benchmark is None:
            testing_version = create_version_name(args, 'selected_sessions')
        else:
            testing_version = create_version_name(args, join(args.benchmark, sep='_'))
        main_version_tester(args, db)
        version_str = f'{args.parameter_name}={current_value} '
        if args.units is not None:
            version_str += args.units
        version_dirs.loc[version_str, 'dir'] = os.path.join(args.result_dir, testing_version)
        version_dirs.loc[version_str, 'value'] = current_value
        replace_config(parameter_rgx, current_value, v, args.result_dir)
        current_value = v

    try:
        version_dirs.sort_values('value', inplace=True)
    except ValueError:
        print('values are not numerical, continuing without sorting')

    if args.unite_benchmarks:
        version_eval(version_dirs.dir.values, version_dirs.index.astype(str),
                     file_name=f'compare_optimize_{args.parameter_name}_' + join(args.benchmark, sep='_'),
                     output_path=args.result_dir, eval_vs=args.eval_vs)
    else:
        for bench in args.benchmark:
            version_eval(version_dirs.dir.values, version_dirs.index.astype(str),
                         file_name=f'compare_optimize_{args.parameter_name}_{bench}',
                         output_path=args.result_dir, eval_vs=args.eval_vs)
    os.chmod(args.result_dir, 0o777)


if __name__ == '__main__':
    optimize_parameter(get_args(), DB())
