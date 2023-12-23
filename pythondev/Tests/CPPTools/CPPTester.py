# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa

from Tests.Utils.PathUtils import create_dir
from Tests.Utils.TesterUtils import setup_logger
from Tests.VSParser import vs_parser
from Tests.vsms_db_api import *

import argparse
import collections
import logging
import os
import platform
import shutil
import subprocess
from time import time, sleep

PYTHONVERSION = 'python3'  # '/usr/bin/python3.7'
SLEEP_TIME = 15
PAR_PROG_LEN = 1500
qsub_jobs = []
parallel_progress = collections.deque(maxlen=PAR_PROG_LEN)


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = vs_parser(description='Neteera vital signs tracker on offline data files')
    parser.add_argument('-cpp', metavar='Location', type=str, required=True, help='location of linux compiled cpp')
    parser.add_argument('-setups', '-session_ids', metavar='ids', nargs='+', required=False, type=int,
                        help='Index of setup in list')
    parser.add_argument('-benchmark', type=str, required=False, help='benchmark on which to run version testing',
                        choices=[bench.value for bench in Benchmark])
    parser.add_argument('--parallel', action='store_true', help='Process setups in parallel')
    return parser.parse_args()


def gen_qsub_job(qsub_iterr, pythondev_dirr, tester_cmdd, output_dirr):
    cur_tester = os.path.join(output_dirr, 'cpp_tester_' + str(qsub_iterr))
    base_str = '#! /bin/sh \n#PBS -l nodes=1 \n#PBS -l walltime=10:30:00\n#PBS -e {}\n' \
        .format(cur_tester + '_log.txt')
    chdir_str = 'cd {}\n'.format(pythondev_dirr)
    no_pycache_str = 'export PYTHONDONTWRITEBYTECODE=1\n'
    with open(cur_tester + '.sh', "w") as text_file:
        text_file.write(base_str)
        text_file.write(chdir_str)
        text_file.write(no_pycache_str)
        text_file.write(tester_cmdd)
    return cur_tester + '.sh'


def delete_qsub_jobs(qsub_jobss):
    pwd = '1234'
    if len(qsub_jobs):
        for job in qsub_jobss:
            del_qsub_job_cmd = '{}/Tests/qdel.sh {}'.format(os.getcwd(), job)
            print(del_qsub_job_cmd)
            subprocess.call('echo {} | sudo -S {}'.format(pwd, del_qsub_job_cmd), shell=True)


def delete_subfolders(output_dir):
    subfolders = [os.path.join(output_dir, o) for o in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, o))]
    for subfolder in subfolders:
        shutil.rmtree(subfolder)


def qsub_jobs_done(args, setupsss, output_dirrr, totall_done, totall, db):
    sleep(SLEEP_TIME)
    dirs = [os.path.join(output_dirrr, res) for res in os.listdir(output_dirrr)
            if os.path.isdir(os.path.join(output_dirrr, res))]
    for cur_out_dir in dirs:
        try:
            handle_result(output_dirrr, cur_out_dir, cur_out_dir[cur_out_dir.rfind('/') + 1:])
        except IndexError:
            continue
    results = [int(res[:res.find('.')]) for res in os.listdir(output_dirrr) if (res.endswith('csv') and
                                                                                res[:res.find('.')].isnumeric() and
                                                                                int(res[:res.find('.')]) in setupsss)]
    par_prog = "Finished: {} out of {}".format(len(results) + totall_done, totall)
    print(par_prog)
    parallel_progress.append(par_prog)
    if len(results) == len(setupsss):
        return True
    else:
        if len(parallel_progress) == PAR_PROG_LEN and all(x == parallel_progress[0] for x in parallel_progress):
            print('no progress in last {} iterations, reinitializing parallel run'.format(PAR_PROG_LEN))
            delete_qsub_jobs(qsub_jobs)
            remaining_setups = []
            for sess in setupsss:
                if sess not in results:
                    remaining_setups.append(sess)
            output_dir = os.path.join(args.result_dir, args.version)
            for qsub_iter, sess_id in enumerate(sorted(remaining_setups)):
                cur_output_dir = os.path.join(args.result_dir, args.version, str(sess_id)) + '/'
                if not handle_output_dir(args.overwrite, cur_output_dir, sess_id):
                    continue
                tester_cmd = gen_cpp_tester_cmd(args, sess_id, cur_output_dir, db)
                path_to_qdel_script = '{}/Tests/qdel.sh'.format(os.getcwd())
                assert os.access(path_to_qdel_script, os.R_OK), 'no permissions to {}, please chmod 777 it...' \
                    .format(path_to_qdel_script)
                assert os.access(path_to_qdel_script, os.W_OK), 'no permissions to {}, please chmod 777 it...' \
                    .format(path_to_qdel_script)
                assert os.access(path_to_qdel_script, os.X_OK), 'no permissions to {}, please chmod 777 it...' \
                    .format(path_to_qdel_script)
                assert os.access(path_to_qdel_script, os.F_OK), 'no permissions to {}, please chmod 777 it...' \
                    .format(path_to_qdel_script)
                path_to_qsub = gen_qsub_job(qsub_iter, os.path.dirname(args.cpp), tester_cmd, output_dir)
                qsub_cmd = 'qsub ' + path_to_qsub
                print(qsub_cmd)
                captured_output = subprocess.run(qsub_cmd, shell=True, capture_output=True).stdout. \
                    decode('utf-8')
                if len(captured_output[:captured_output.find('.')]):
                    qsub_jobs.append(captured_output[:captured_output.find('.')])
                sleep(0.2)
            parallel_progress.clear()
        return False


def get_config(idx, db):
    try:
        mount = db.setup_mount(idx).lower()
        posture = db.setup_posture(idx).lower()
        target = db.setup_target(idx).lower()
    except AttributeError as e:
        print(e)
        logging.getLogger('vsms').warning(f'setup {idx} invalid mounting: using default (seat front)')
        return 'cf'
    if mount == 'seat_back':
        config = 'cb'
    elif mount == 'bed_bottom':
        config = 'bu'
    elif mount in ['bed_top', 'lab_ceiling'] \
            or (mount == 'lab_wall' and posture == 'lying'):    # bed side config is same as bed top
        config = 'ba'
    elif mount in ['tripod', 'lab_wall'] and posture == 'sitting' and target == 'front':
        config = 'cf'
    elif posture == 'standing' and target == 'front':
        config = 'sf'
    else:
        print(f'setup {idx} invalid mounting: using default (seat front)')
        config = 'cf'
    return config


def gen_compute_str(argsss):
    compute_string = ''
    all_cpp_vs = ['hr', 'rr', 'ra', 'stat', 'ie', 'hrv']
    comp_vs = [argsss.compute[vs_iter].name for vs_iter in range(len(argsss.compute))]
    for vs in all_cpp_vs:
        if vs not in comp_vs:
            if vs == 'stat':
                compute_string += ' -no_cs'
            else:
                compute_string += ' -no_' + vs
    return compute_string


def gen_cpp_tester_cmd(argss, sess_id, cur_output_dir, db):
    compute_str = gen_compute_str(argss)
    tester_cmd = f'{argss.cpp} -no_lag -a -session {sess_id} -results_dir ' + cur_output_dir + compute_str\
                 + ' -db -sensor fmcw_mockup -setup ' + get_config(sess_id, db)
    print(tester_cmd)
    return tester_cmd


def handle_result(out_dir, cur_out_dir, idx):
    lines = []
    if len(os.listdir(cur_out_dir)):
        res_fname = ''
        for fn in os.listdir(cur_out_dir):
            if fn.startswith('results_'):
                res_fname = os.path.join(cur_out_dir, fn)
        if os.path.isfile(res_fname):
            with open(res_fname) as f:
                for cnt, line in enumerate(f):
                    lines.append(line)
            if len(lines) > 2:
                if '*,*,*,*,*,*' in lines[-1]:
                    # results csv file is written continuously. **** are appended to indicate that the setup is done
                    with open(res_fname, "w") as f:
                        for line in lines[:-1]:
                            f.write('%s' % line)
                    shutil.copy2(res_fname, os.path.join(out_dir, str(idx)) + '.csv')
                    summary_dest = out_dir.replace('dynamic', 'spot')
                    if not os.path.isdir(summary_dest):
                        os.mkdir(summary_dest)
                    shutil.copy2(res_fname.replace('results_', 'sessionSummary_'),
                                 os.path.join(summary_dest, str(idx)) + '.csv')


def setup_done(idxx, out_filee, check_ending=True):
    lines = []
    with open(out_filee) as f:
        for cnt, line in enumerate(f):
            lines.append(line)
    if check_ending:
        if len(lines) > 2:
            if '*,*,*,*,*,*' in lines[-1]:
                print('Skipped: {}'.format(idxx))
                return True
    else:
        print('Skipped: {}'.format(idxx))
        return True
    return False


def handle_output_dir(overwrite, out_dir, idx):
    if os.path.isdir(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
            try:
                os.remove(out_dir[:-1] + '.csv')
            except FileNotFoundError:
                pass
            os.makedirs(out_dir)
            os.system('chmod 777 -R ' + out_dir)
        else:
            out_file = os.path.abspath(os.path.join(out_dir, '../..')) + '/' + str(idx) + '.csv'
            if os.path.isfile(out_file):
                return setup_done(idx, out_file)
    else:
        os.makedirs(out_dir)
        os.system('chmod 777 -R ' + out_dir)
    return False


def remove_finished_setups(idss, output_dirr):
    remaining_setups = []
    for idd in idss:
        cur_out_file = os.path.join(output_dirr, str(idd)) + '.csv'
        if os.path.isfile(cur_out_file) and setup_done(idd, cur_out_file, check_ending=False):
            continue
        else:
            remaining_setups.append(idd)
    os.system('chmod 777 -R ' + output_dirr)
    return remaining_setups


def main(args: argparse.Namespace) -> int:
    db = DB()
    output_dir = os.path.join(args.result_dir, args.version, 'dynamic')
    create_dir(output_dir)
    os.system('chmod 777 -R ' + output_dir)
    if args.benchmark:
        ids = db.benchmark_setups(args.benchmark)
        total = len(ids)
    elif args.setups:
        total = len(args.setups)
        ids = args.setups
    else:
        raise ValueError('setups or benchmark is required')
    setup_logger(args.result_dir, args.silent, ids)
    if (args.setups or args.benchmark) and not args.overwrite:
        ids = sorted(remove_finished_setups(ids, output_dir))
    else:
        ids = sorted(ids)
        logging.getLogger('vsms').info('Overwrite existing files\r\n')
    total_done = total - len(ids)
    if len(ids):
        logging.getLogger('vsms').warning('Evaluating: {}\r\n'.format(str(ids)))
    else:
        logging.getLogger('vsms').warning('All setups have been processed')
    start_time = time()
    if args.parallel:
        if platform.system() == 'Linux':  # parallel linux - qsub runs
            for qsub_iter, sess_id in enumerate(ids):
                cur_output_dir = os.path.join(output_dir, str(sess_id)) + '/'
                if handle_output_dir(args.overwrite, cur_output_dir, sess_id):
                    continue
                tester_cmd = gen_cpp_tester_cmd(args, sess_id, cur_output_dir, db)
                path_to_qsub = gen_qsub_job(qsub_iter, os.path.dirname(args.cpp), tester_cmd, output_dir)
                qsub_cmd = 'qsub ' + path_to_qsub
                print(qsub_cmd)
                captured_output = subprocess.run(qsub_cmd, shell=True, capture_output=True).stdout. \
                    decode('utf-8')
                if len(captured_output[:captured_output.find('.')]):
                    qsub_jobs.append(captured_output[:captured_output.find('.')])
            while not qsub_jobs_done(args, ids, output_dir, total_done, total, db):
                sleep(SLEEP_TIME)
        else:
            print('parallel run not supported in windows environment')
    else:  # serial
        for sess_id in ids:
            cur_output_dir = os.path.join(output_dir, str(sess_id)) + '/'
            if handle_output_dir(args.overwrite, cur_output_dir, sess_id):
                continue
            os.chdir(os.path.dirname(args.cpp))
            tester_cmd = gen_cpp_tester_cmd(args, sess_id, cur_output_dir, db)
            subprocess.run(tester_cmd, shell=True)
            try:
                handle_result(output_dir, cur_output_dir, sess_id)
            except IndexError:
                continue
    if round(time() - start_time) > 0:
        print('Total algorithm runtime: {} Seconds'.format(round(time() - start_time)))
    return 0


if __name__ == "__main__":
    bargs = get_args()
    assert bargs.cpp.endswith('live_main') or os.path.isfile(bargs.cpp), 'live_main doesnt exists, aborting..\r\n'
    try:
        main(bargs)
    finally:
        if bargs.parallel and platform.system() == 'Linux':
            print('removing all qsub jobs')
            delete_qsub_jobs(qsub_jobs)
            # print('deleting all folders in output directory')
            # delete_subfolders(os.path.join(bargs.result_dir, bargs.version, 'dynamic'))
