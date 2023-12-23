import logging
import subprocess
from io import StringIO
from subprocess import PIPE
from time import strftime, gmtime

import numpy as np
import pandas as pd

from Tests.Utils.StringUtils import join
from Tests.Utils.PathUtils import create_dir
import os


def gen_tester_cmd(argss, setups, ver_name, parallel=None):
    # in qsub parallel, can't use multiprocessing, as all jobs MUST be managed by qsub

    test_command = f'python3 ./Tests/Tester.py -version {ver_name} -result_dir {argss.result_dir} -compute' \
                   f' {join(argss.compute)} -setups  {join(setups)}'
    if 'cpp' in argss.__dict__.keys() and argss.cpp is not None:
        test_command = test_command.replace('Tester', os.path.join('CPPTools', 'CPPTester')) + f' -cpp {argss.cpp}'
    if parallel:
        test_command += ' --parallel'
    if argss.overwrite:
        test_command += ' --overwrite'
    if argss.force:
        test_command += ' --force'
    return test_command


def setup_logger(log_path: str, silent: bool, setups) -> None:
    """ Set up logger in log path

    :param str log_path: directory for log file
    :param bool silent: False to log debug-level and info-level logs
    :param str setups: lists of setups
    """
    base_filename = setups[0] if len(setups) == 1 else 'main'
    create_dir(log_path)
    timestamp = ''.join(strftime("%d_%b_%Y_%H_%M_%S", gmtime())).replace(' ', '_')
    logfile = os.path.join(log_path, f'{base_filename}_{timestamp}_logs.txt')
    ch = logging.StreamHandler()
    level = logging.WARNING if silent else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)-15s %(message)s',  filemode='w', filename=logfile)
    ch.setLevel(level)
    logging.getLogger('vsms').addHandler(ch)
    logging.getLogger('vsms').info('started logging')


def proc_output_to_df(cmd_output, skip_rows):
    string_data = StringIO(cmd_output.decode('utf-8'))
    return pd.read_csv(string_data, skiprows=skip_rows)


def get_qstat_df():
    proc = subprocess.Popen('qstat', shell=True, stdout=PIPE)
    output = proc.communicate()
    if len(output[0]) == 0:
        return pd.DataFrame({'id': [], 'Queue': [], 'S': []})
    try:
        df = proc_output_to_df(output[0], [1])
        columns = df.columns[0].split()[1:]
        if len(columns) == 0:
            df = proc_output_to_df(output[0], [0])
            columns = df.columns[0].split()[1:]
    except (pd.errors.EmptyDataError, ):  # q_stat is empty
        return pd.DataFrame({'id': [], 'Queue': [], 'S': []})
    try:
        del columns[4]
    except IndexError:
        return pd.DataFrame({'id': [], 'Queue': [], 'S': []})
    new_df = pd.DataFrame(columns=columns)
    for i, row in enumerate(df.values):
        new_df.loc[i] = row[0].split()
    return new_df


def get_running_jobs():
    for _ in range(10):
        try:
            q_stat = get_qstat_df()
            q_stat = q_stat[np.logical_and.reduce((q_stat.Queue != 'gpu', q_stat.S != 'C', q_stat.S != 'E'))]
            return [pbs_id.split('.')[0] for pbs_id in q_stat.id.values]
        except OSError:
            pass