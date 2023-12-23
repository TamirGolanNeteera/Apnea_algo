import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__))) # noqa

from Tests.CPPTools.BitExactPythonCPP import check_bit_exact
from Tests.Utils.PathUtils import create_dir, windows_dir_to_linux_dir, copy_code_and_change_working_dir
from Tests.vsms_db_api import *
from Tests.Utils.TestsUtils import run_cmd, create_version_name, intersect, last_delivered_version
from Tests.Utils.TesterUtils import gen_tester_cmd
from Tests.VSParser import *
from Tests.Tester import tester_main
from Tests.Evaluation.RangeEstimationEvaluator import main_range_evaluator
from Tests.Constants import DELIVERED

import argparse
import os
import copy


def get_parser():
    """get parser and VersionTester arguments, can be used by other scripts with more arguments"""
    parser = vs_parser(description='Test a specific python code version.')
    parser.add_argument('-setups', '-session_ids', '-setup_ids', '-setup', metavar='ids', nargs='+', type=int,
                        help='Index of session in list OR setups in DB', required=False)
    parser.add_argument('--plot_res', '--plot', '--plot_ref', '--plot_nes', action='store_true',
                        help='plot nes-ref per session')
    parser.add_argument('--plot_all', action='store_true', help='plot freq-time, psd plots per session')
    parser.add_argument('-cpp', '-cpp_path', '-live_main_path', type=str, required=False,
                        help='location of linux compiled cpp')
    parser.add_argument('-benchmark', '-benchamrk', '-benchmarks', '-benchamrks', type=str, nargs='+',
                        choices=[bench.value for bench in Benchmark] + [project.value for project in Project],
                        help='benchmark on which to run version testing, see vsms_db_api.py for more information')
    parser.add_argument('--no_reliability', action='store_true', help='Dont take reliability into account')
    parser.add_argument('--unite_benchmarks', action='store_true',
                        help='run on all the benchmarks in one folder')
    parser.add_argument('-compare_to', type=str, help='delivered version to compare to, default is the latest',
                        nargs='+')
    parser.add_argument('--dont_evaluate', action='store_true', help='do not run evaluation scripts')
    parser.add_argument('--only_compare', '--compare_only', action='store_true',
                        help='do not run evaluation scripts except comparison')
    parser.add_argument('--fast_run', action='store_true',
                        help='run fewer sessions for back, no sessions under the bed')
    parser.add_argument('-eval_vs', type=str, nargs='+', help='show plot in the end of the session',
                        choices=VS_SUPPORTED, required=False)
    return parser


def get_args() -> argparse.Namespace:
    return post_parsing(get_parser().parse_args())


def run_tester(argss, ids, testing_ver):
    """run the given sessions"""
    if argss.cpp is not None:
        argss.cpp = windows_dir_to_linux_dir(argss.cpp)
        assert platform.system() == 'Linux', 'CPP version testing is available only in linux\r\n'
        assert os.access(argss.cpp, os.X_OK), 'no permission to live_main folder'
        if not os.path.isfile(os.path.join(os.path.dirname(argss.cpp), 'VitalSignsMonitoring.cfg')):
            if 'live_main' in os.listdir(argss.cpp):
                argss.cpp = os.path.join(argss.cpp, 'live_main')
        test_command = gen_tester_cmd(argss, ids, testing_ver, True)
        run_cmd(test_command)
    else:
        args_for_tester = copy.deepcopy(argss)
        args_for_tester.setups = ids
        args_for_tester.parallel = True
        args_for_tester.profile = False
        args_for_tester.version = testing_ver
        tester_main(args_for_tester)
    dynamic_path = os.path.join(argss.result_dir, testing_ver, 'dynamic')
    if not os.path.exists(dynamic_path) or not is_dir_chmoded(dynamic_path):
        os.system('chmod 777 -R ' + argss.result_dir)


def is_dir_chmoded(path):
    mode = oct(os.stat(path).st_mode)
    return int(mode[-1]) >= 7


def run_plotter(argss, output_folder, ids, plot_vs):
    """ run PlotResults.py and PlotRawDataRadarCPX.py"""
    plot_cmd = ' ./Tests/Plots/PlotResults.py --silent  -result_dir ' + output_folder
    if plot_vs is not None and len(plot_vs) < len(VS_SUPPORTED):
        plot_cmd += os.sep + 'dynamic'
        plot_cmd += f' -vital_sign {join(plot_vs)}'
    if argss.force:
        plot_cmd += ' --force'
    run_cmd(plot_cmd)
    if argss.plot_all:
        plot_raw_data_command = ' ./Tests/Plots/PlotRawDataRadarCPX.py -ppt plot_raw_data_hr -save_path ' + \
                                os.path.join(output_folder, 'raw_data_plots') + f' --dont_show -setups {join(ids)}'
        run_cmd(plot_raw_data_command)


def version_eval(output_folders, ver_names=(), file_name=None, output_path=None, output_type=None, eval_vs=None,
                 force=None, merge_benchmarks=False):
    """runs VersionEvaluator.py"""
    folder_str = join(output_folders)
    names_str = join(ver_names)
    cmd_eval = f' ./Tests/Evaluation/VersionEvaluator.py -folder_list {folder_str}'

    if len(ver_names):
        cmd_eval += ' -version_names ' + names_str
    if output_path is not None:
        cmd_eval += ' -output_path ' + output_path
    if output_type is not None:
        cmd_eval += ' -outputs ' + output_type
    if file_name is not None:
        cmd_eval += f' -fname {file_name}'
    if eval_vs is not None:
        cmd_eval += f' -vital_signs {join(eval_vs)}'
    if force:
        cmd_eval += ' --force'
    if merge_benchmarks:
        cmd_eval += ' --merge_benchmarks'
    run_cmd(cmd_eval)


def get_previous_version_dir(ver: str, ben, ending):
    if os.path.isdir(ver):
        return ver
    if os.path.isdir(os.path.join(DELIVERED, ver, 'stats', ben, ending)):
        return os.path.join(DELIVERED, ver, 'stats', ben, ending)
    temp_ver = ver[:-1] if ver.endswith(os.sep) else ver
    if os.path.isdir(f'{temp_ver}_{ben}'):
        return f'{temp_ver}_{ben}'
    raise ValueError(f'previous version {ver} not found')


def compare_version(output_folder, ben, is_spot, prev_vers, db, sessions=(), cpp=False, output=None, eval_vs=None):
    """compare 2 or more versions, runs VersionEvaluator.py"""
    if prev_vers is None:
        prev_vers = [last_delivered_version()]
    if ben not in [str(x) for x in Benchmark]:
        if not intersect([sessions, db.benchmark_setups(Benchmark.fae_rest)]):
            return
        ben = 'fae_rest'
    list_dir = os.listdir(output_folder)
    if 'spot' in list_dir or 'dynamic' in list_dir:
        ending = ''
    elif is_spot:
        ending = 'spot'
    else:
        ending = 'dynamic'

    output_folders = [output_folder]
    for ver in prev_vers:
        output_folders.append(get_previous_version_dir(ver, ben, ending))
    if cpp:
        check_bit_exact(*output_folders, 0.01)
    try:
        version_eval(output_folders, output_type=output, eval_vs=eval_vs)
    except FileNotFoundError as e:
        print(e)
        print('no result for benchmark in the latest version')


def test_version_on_sessions(args, db, benchmark=None, sessions=None):
    """tests the version (run, plot, compare, evaluate...) for 1 benchmark or list of sessions"""
    testing_version = create_version_name(args, benchmark)
    output_dir = os.path.join(args.result_dir, testing_version)
    create_dir(output_dir)
    os.chmod(output_dir, 0o777)
    copy_code_and_change_working_dir(output_dir)
    args.log_path = output_dir
    run_tester(args, sessions, testing_version)

    if not args.only_compare and not args.dont_evaluate:
        version_eval([output_dir], [testing_version], eval_vs=args.eval_vs)

    if args.plot_res or args.plot_all:
        run_plotter(args, output_dir, sessions, args.eval_vs)

        if not args.spot_mode:
            try:
                main_range_evaluator(output_dir, 150, args.cpp)
            except ZeroDivisionError as e:  # empty dataframe
                print(e)
                pass
    if not args.dont_evaluate:
        compare_version(output_dir, benchmark.replace('_back_front', ''), args.spot_mode, args.compare_to, db, sessions,
                        args.cpp, eval_vs=args.eval_vs)
    return output_dir


def benchmark_broad(benchmark, db):
    if benchmark in [str(x) for x in Benchmark]:
        return db.benchmark_setups(benchmark)
    if benchmark in [str(x) for x in Project]:
        return db.setup_by_project(benchmark)


def gen_sessions_ids(benchmarks, db, fast_run=False):
    benchmark_lists = [benchmark_broad(bench, db) for bench in benchmarks]
    sessions = sorted(list(set().union(*benchmark_lists)))
    if fast_run:
        bed_bottom = db.setup_by_mount(Mount.bed_bottom)
        szmc = db.benchmark_setups(Benchmark.szmc_clinical_trials)
        standing = db.setup_by_posture(Posture.standing)
        seat_back = db.setup_by_mount(Mount.seat_back)
        stress = db.setup_by_note('stress')
        sessions = set(sessions) - set(bed_bottom + standing + seat_back) | set(szmc + stress)
    return sessions


def main_version_tester(args, db):
    args.plot_res = args.plot_res or args.plot_all
    if not (args.setups or args.benchmark):
        args.benchmark = ['szmc_clinical_trials']
    if args.benchmark is None:
        return test_version_on_sessions(args, db, 'selected_sessions', args.setups)
    else:
        if args.unite_benchmarks:
            return test_version_on_sessions(args, db, '_'.join(args.benchmark), gen_sessions_ids(args.benchmark, db))
        else:
            for bench in args.benchmark:
                sess_list = gen_sessions_ids([bench], db, args.fast_run)
                return test_version_on_sessions(args, db, bench, sess_list)


if __name__ == "__main__":
    out_dir = main_version_tester(get_args(), DB())
    os.system('chmod 777 -R ' + out_dir)
