from distutils.dir_util import copy_tree

from Tests.Plots.PlotResults import main_plot_results, run_cmd
from Tests.Utils.DBUtils import calculate_delay
from Tests.Utils.PathUtils import create_dir
from Tests.Utils.TestsUtils import collect_result, intersect, last_delivered_version
from Tests.Utils.StringUtils import join
from Tests.VersionTester import post_parsing, get_parser, argparse, version_eval
from Tests.Constants import DELIVERED
from Tests.vsms_db_api import *

import datetime


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = get_parser()
    parser.add_argument('-days', type=float, help='number of recent fays to take setups from', default=2)
    parser.add_argument('-save_fig_path', metavar='Location', type=str, required=False,
                        help='location of output figures')
    parser.add_argument('--diff', action='store_true', help='Plot the pred-ref differences')
    parser.add_argument('-ppt_fname', type=str, required=False, help='Powerpoint filename to generate ppt '
                                                                     'presentation from plots')

    parser.add_argument('-product', type=str, required=False, default='health', help='automotive or health')
    parser.add_argument('-vital_sign', nargs='+', type=str, choices=['hr', 'rr', 'bbi', 'stat'],
                        help='plot the following vital signs (default is all)\n',
                        default=['hr', 'rr', 'bbi', 'stat'])
    parser.add_argument('--csv', action='store_true', help='vital sign tracker results from csv files instead of npy \
     files')
    parser.add_argument('--t0', action='store_true', help='start each setup from its t0')
    parser.add_argument('-match_list_type', metavar='match_lists', type=str, required=False,
                        choices=['NO', 'MSE', 'TS'], default='TS', help='what type of match list to use')
    parser.add_argument('-company', metavar='company', type=str, choices=[project.value for project in Project],
                        required=False, help='what company to look for')
    parser.add_argument('-mount', type=str, choices=[mount.value for mount in Mount],
                        required=False, help='what company to look for')

    return post_parsing(parser.parse_args())


if __name__ == "__main__":
    args = get_args()
    db = DB()
    cwd = os.getcwd()
    now = datetime.datetime.now()

    setups = db.setup_from_to_date(from_date=now - datetime.timedelta(days=args.days), to_date=now)
    if args.company is not None:
        setups = intersect([db.setup_by_project(args.company), setups])
    if args.mount is not None:
        setups = intersect([db.setup_by_mount(args.mount), setups])
    setups = [x for x in setups if os.path.exists(db.setup_ref_path(setup=x, sensor=Sensor.nes)[0])]

    result_dir = args.result_dir + now.strftime("%d_%m_%Y")
    args.result_dir = result_dir

    create_dir(args.result_dir)
    collect_result(setups, os.path.join(args.result_dir, 'online'))
    if args.compare_to is not None:
        for i, ver in enumerate(args.compare_to):
            if ver == 'last':
                ver = last_delivered_version()
            pythondev_dir = ver if os.path.isdir(ver) else os.path.join(DELIVERED, ver, ver)
            os.chdir(pythondev_dir)
            print(f'working dir changed to {os.getcwd()}')
            cmd = f' ./Tests/VersionTester.py -version {ver} -result_dir {result_dir}'
            if 'mc-py' in pythondev_dir:
                cmd += f'{os.sep}{ver}{os.sep}{os.sep} -log_path {result_dir} -session_ids {join(setups)} -seed 13 -compute hr rr'
                run_cmd(cmd + ' spot_hr ')
                run_cmd(cmd + ' --spot_mode')
                for dir_name in os.listdir(os.path.join(result_dir, ver)):
                    full_dir = os.path.join(result_dir, ver, dir_name)
                    if os.path.isdir(full_dir):
                        dest_dir = os.path.join(result_dir, ver, 'spot' if '_sm' in dir_name else 'dynamic')
                        os.rename(full_dir, dest_dir)
            else:
                cmd += f' --dont_evaluate -setups {join(setups)} '
                run_cmd(cmd)
    os.chdir(cwd)
    for folder_ver in os.listdir(result_dir):
        if os.path.isdir(folder_ver):
            ver_full_path = os.path.join(result_dir, folder_ver)
            if 'dynamic' not in os.listdir(ver_full_path):
                copy_tree(ver_full_path, os.path.join(ver_full_path, 'dynamic'))
    main_plot_results(args, db)
    version_eval([result_dir], force=args.force)
