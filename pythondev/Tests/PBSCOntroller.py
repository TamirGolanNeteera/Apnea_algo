# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
import sys

from Tests.VSParser import *
from Tests.Utils.TestsUtils import run_cmd
from Tests.Utils.TesterUtils import gen_tester_cmd, get_running_jobs

from time import sleep
import numpy as np
import subprocess
import copy

PYTHONVERSION = 'python3'  # '/usr/bin/python3.7'
MAX_JOBS = 1e5
SLEEP_TIME = 5


def setups_done(setup_ids, folder):
    if os.path.isdir(os.path.join(folder, 'spot')):
        results = np.unique([int(res[:res.find('_')]) for res in os.listdir(os.path.join(folder, 'spot'))
                             if res[:res.find('_')].isdigit() and (res.endswith('.npy') or res.endswith('.data'))
                             and int(res[:res.find('_')]) in setup_ids])
    else:
        results = []
    return results


class JobsController:
    def __init__(self, args, out_dir):
        self.qsub_jobs = {}
        self.qsub_iter = 0
        self.re_submitted_counter = {}
        self.args = args
        self.out_dir = out_dir
        self.qsub_dir = os.path.join(out_dir, 'qsub_tester_jobs')
        self.finished_jobs = -1     # start as -1 so 0 will be printed
        self.died_setups_ids = {}
        self.last_printed = ''
        self.same_print_counter = 0
        try:
            os.mkdir(self.qsub_dir)
        except FileExistsError:
            pass

    def qsub_jobs_done(self, ids, tot):
        finished_jobs = len(setups_done(ids, self.out_dir))
        if 'Finished: ' in self.last_printed:
            sys.stdout.write('\b' * 50 + f'Finished: {finished_jobs} out of {tot} X{self.same_print_counter + 1}')
        else:
            self.same_print_counter = 0
            self.print(f'Finished: {finished_jobs} out of {tot}')
        self.same_print_counter += 1
        return (finished_jobs == len(ids)), finished_jobs

    def run_parallel(self, ids, total):
        qsub_done, finished_jobs = self.qsub_jobs_done(ids, total)
        if not qsub_done:
            pythondev_folder = os.getcwd()
            self.submit_jobs(ids, pythondev_folder)
            while len(self.died_setups_ids):
                self.submit_jobs(self.died_setups_ids, pythondev_folder)
            self.finished_jobs = -1
            while not qsub_done:
                sleep(SLEEP_TIME)
                qsub_done, finished_jobs = self.qsub_jobs_done(ids, total)
                self.re_submit_jobs(pythondev_folder)
                if len(self.qsub_jobs) == 0:
                    self.submit_jobs(set(ids) - set(setups_done(ids, self.out_dir)), pythondev_folder)

    def submit_jobs(self, ids, pythondev_dir):
        self.print(f'{len(ids)} job submitted')
        for i, setup_id in enumerate(list(ids)):
            self.died_setups_ids.pop(setup_id, None)
            if VitalSignType.bbi in self.args.compute:
                while len(self.qsub_jobs) > 300:
                    self.remove_finished_setups(setups_done(list(self.qsub_jobs.keys()), self.out_dir))
                    sleep(1)
            else:
                if len(self.qsub_jobs) > 200:
                    self.remove_finished_setups(setups_done(list(self.qsub_jobs.keys()), self.out_dir))
            tester_cmd = gen_tester_cmd(self.args, [setup_id], self.args.version, False)
            if not self.args.silent or self.qsub_iter % 50 == 0:
                self.print(tester_cmd)
            path_to_qsub = self.gen_qsub_job(pythondev_dir, tester_cmd, setup_id)
            qsub_cmd = 'qsub ' + path_to_qsub
            sleep(max(0.2, len(self.qsub_jobs) / 2500))
            if not self.args.silent or self.qsub_iter % 50 == 1:
                self.print(qsub_cmd)
            captured_output = subprocess.run(qsub_cmd, shell=True, capture_output=True).stdout.decode('utf-8')
            if len(captured_output[:captured_output.find('.')]):
                self.qsub_jobs[setup_id] = captured_output[:captured_output.find('.')]

    def remove_finished_setups(self, finished_setups):
        for setup in finished_setups:
            self.qsub_jobs.pop(setup, None)
        running = set(get_running_jobs())
        for setup, job in copy.deepcopy(self.qsub_jobs).items():
            if setup not in finished_setups and job not in running:
                self.re_submitted_counter[setup] = 1 + self.re_submitted_counter.get(setup, 0)
                self.qsub_jobs.pop(setup, None)
                self.died_setups_ids[setup] = 1
                if self.re_submitted_counter[setup] > 3:
                    self.run_setup_serial_mode(setup)

    def run_setup_serial_mode(self, se):
        self.print(f'setup {se} submitted three times and concluded without a result file, running it serial')
        self.args.parallel = False
        self.args.overwrite = False
        self.args.setup_ids = [se]
        self.args.silent = False
        run_cmd(gen_tester_cmd(self.args, [se], self.args.version, False))

    def re_submit_jobs(self, pythondev_dir):
        """fixer for a bug of dying jobs in PBS"""
        assert len(self.qsub_jobs) < MAX_JOBS, f'more than {MAX_JOBS} submitted, crushing'
        finished = setups_done(list(self.qsub_jobs.keys()), self.out_dir)
        self.remove_finished_setups(finished)
        running_ids = get_running_jobs()
        for setup in list(self.died_setups_ids)[:150 - len(running_ids)]:
            if setup not in self.qsub_jobs or self.qsub_jobs[setup] in running_ids:
                continue
            self.print(f're-submitting: ', end='')
            if self.re_submitted_counter.get(setup, 0) >= 2:
                self.run_setup_serial_mode(setup)
            self.re_submitted_counter[setup] = 1 + self.re_submitted_counter.get(setup, 0)
            if setup not in finished:
                self.finished_jobs = -1
                self.print(setup, end=' ')
                self.submit_jobs([setup], pythondev_dir)
                sleep(1)

    def gen_qsub_job(self, python_dir, cmd, setup):
        cur_tester = os.path.join(self.qsub_dir, f'tester_{self.qsub_iter}_setup_{setup}')
        self.qsub_iter += 1
        base_str = '#! /bin/sh' \
                   '\n#PBS -l nodes=1' \
                   '\n#PBS -l walltime=02:30:00' \
                   f'\n#PBS -e {cur_tester}_log.txt' \
                   '\n'
        chdir_str = 'cd {}\n'.format(python_dir)
        with open(cur_tester + '.sh', "w") as text_file:
            text_file.write(base_str)
            text_file.write(chdir_str)
            text_file.write(cmd)
        return cur_tester + '.sh'

    def print(self, string, **kwargs):
        if 'Finished' in string:
            print('\n')
        print(string, **kwargs)
        self.last_printed = string
