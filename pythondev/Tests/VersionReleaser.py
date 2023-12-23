# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
from Tests.Constants import DELIVERED
from Tests.Evaluation.BugsTester import main_bug_tester
from Tests.VersionTester import get_parser as ver_test_parser, post_parsing, main_version_tester, compare_version, \
    version_eval
from Tests.Utils.TestsUtils import run_cmd, intersect
from Tests.Utils.PathUtils import create_dir
from Tests.send_mail import mail_template
from Tests.vsms_db_api import DB, Benchmark

from pylibneteera.datatypes import VitalSignType

from distutils.dir_util import copy_tree
from time import sleep
import shutil
import copy
import os

BENCHMARKS = ['szmc_clinical_trials', 'cen_exel', 'ec_benchmark', 'fae_rest', 'mild_motion', 'nwh', 'fae_special']
BENCHMARKS_TO_MERGE = ['fae_rest', 'mild_motion']

VITAL_SIGNS = [VitalSignType.hr, VitalSignType.rr, VitalSignType.ra, VitalSignType.ie, VitalSignType.stat,
               VitalSignType.bbi, VitalSignType.identity]
VITAL_SIGNS_EC = [VitalSignType.hr, VitalSignType.rr, VitalSignType.stat]


def get_args():
    parser = ver_test_parser()
    parser.add_argument('-path_to_version', type=str, help='path to version code', required=False)
    return post_parsing(parser.parse_args())


def run_single_bench(bench, ver_name, result_dir):
    args_for_ver_tester = copy.deepcopy(args)
    args_for_ver_tester.compute = VITAL_SIGNS_EC if bench in ['ec_benchmark', 'mild_motion'] else VITAL_SIGNS
    args_for_ver_tester.spot_mode = False
    args_for_ver_tester.result_dir = result_dir
    args_for_ver_tester.benchmark = [bench]
    args_for_ver_tester.version = ver_name
    args_for_ver_tester.dont_evaluate = True
    main_version_tester(args_for_ver_tester, db)
    saved_path = os.path.join(args_for_ver_tester.result_dir, f'{ver_name}_{bench}')
    new_saved_path = os.path.join(args_for_ver_tester.result_dir, bench)
    os.rename(saved_path, new_saved_path)


def compared_merged_branches(versions):
    folders = [os.path.join(DELIVERED, ver, 'stats', x) for x in BENCHMARKS_TO_MERGE for ver in versions]
    version_eval(folders, merge_benchmarks=True, output_type='xlsx')


def run_benchmarks(version_name, prev_version_name, benchs):
    """run benchmarks, copy results to delivered and compare to prev version"""
    os.chdir(os.path.join(DELIVERED, version_name, version_name))
    result_dir = os.path.join(DELIVERED, version_name, 'stats')
    for bench in benchs:
        run_single_bench(bench, version_name, result_dir)
        compare_version(os.path.join(result_dir, bench), bench, True, prev_version_name, db, output='xlsx')
    compared_merged_branches([version_name, prev_version_name])
    run_plot_results(version_name)
    for bench in benchs:
        compare_version(os.path.join(result_dir, bench), bench, True, prev_version_name, db, output='pptx')


def gen_delivered_dir(version_name, path_to_version):
    if path_to_version is None:
        return
    stats = os.path.join(DELIVERED, version_name, 'stats')
    if os.path.isdir(os.path.join(DELIVERED, version_name)):
        if os.path.isdir(stats) and len(intersect([os.listdir(stats), [str(x) for x in Benchmark]])):
            raise FileExistsError(f'folder {os.path.join(DELIVERED, version_name)} already exists, aborting...')
        print('overwriting')
        if os.path.isdir(stats):
            shutil.rmtree(stats, ignore_errors=False, onerror=None)
    create_dir(os.path.join(DELIVERED, version_name, version_name))
    copy_tree(path_to_version, os.path.join(DELIVERED, version_name, version_name))
    sleep(1)
    all_dirs = [x[0] for x in os.walk(os.path.join(DELIVERED, version_name, version_name))]
    all_files = []
    for d in all_dirs:
        if os.path.basename(d).startswith('.') or d.endswith('__pycache__'):
            shutil.rmtree(d, ignore_errors=False, onerror=None)
        else:
            try:
                all_files.extend([os.path.join(d, dd) for dd in os.listdir(d)])
            except FileNotFoundError:
                pass
    for f in all_files:
        if os.path.basename(f).startswith('.'):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    shutil.copy2(os.path.join(DELIVERED, version_name, version_name, 'Version_changes.xls'),
                 os.path.join(DELIVERED, version_name, f'{version_name}_Version_changes.xls'))


def run_plot_results(ver):
    plots_cmd = ' ./Tests/Plots/PlotResults.py -result_dir ' + os.path.join(DELIVERED, ver, 'stats')
    run_cmd(plots_cmd)


if __name__ == '__main__':
    db = DB()
    args = get_args()
    gen_delivered_dir(args.version, args.path_to_version)

    os.system('chmod 777 -R ' + os.path.join(DELIVERED, args.version))
    benchmarks = BENCHMARKS if args.benchmark is None else args.benchmark
    run_benchmarks(args.version, args.compare_to, benchmarks)
    mail_template(args.version)
    os.system(f"python3 ./Tests/VersionTester.py -result_dir {os.path.join(DELIVERED, args.version, 'Tests')}"
              "-version checking_cpp_run -benchmark ec_benchmark"
              "-cpp /Neteera/Work/homes/nachum_shtauber/Neteera/VER-1_23_2_0/live_main -compute hr rr stat ra ie")
    main_bug_tester(args)
    os.system('chmod 777 -R ' + os.path.join(DELIVERED, args.version))
