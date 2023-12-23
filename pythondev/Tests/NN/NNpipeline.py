import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa  # noqa
from sklearn.model_selection import train_test_split
from Tests.VersionTester import *

PYTHONVERSION = 'python3'


def gen_tester_cmd(argss, sessionss, testing_versionn, parallel=None, seed=None):
    # in qsub parallel, can't use multiprocessing, as all jobs MUST be managed by qsub
    if 'cpp' in argss.__dict__.keys() and argss.cpp is not None:
        tester_command = PYTHONVERSION + ' ./Tests/CPPTester.py -version ' + testing_versionn + ' -log_path ' \
                         + argss.log_path + ' -result_dir ' + argss.result_dir + ' -compute ' \
                         + ' '.join([vs.name for vs in argss.compute]) \
                         + ' -session_ids ' + ' '.join(list(map(str, map(int, sessionss)))) + ' -cpp ' + argss.cpp
    else:
        tester_command = PYTHONVERSION + ' ./Tests/Tester.py -version ' + testing_versionn + ' -log_path ' \
                         + argss.log_path + ' -result_dir ' + argss.result_dir + ' -compute ' \
                         + ' '.join([vs.name for vs in argss.compute]) \
                         + ' -session_ids ' + ' '.join(list(map(str, map(int, sessionss))))
        if argss.silent:
            tester_command += ' --silent'
        if argss.spot_mode:
            tester_command += ' --spot_mode'
    if seed:
        tester_command += ' -seed {}'.format(seed)
    if parallel:
        tester_command += ' --parallel'
    if argss.overwrite:
        tester_command += ' --overwrite'
    return tester_command


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--spot_mode', action='store_true', help='Run tester in spot mode')
    parser.add_argument('--parallel', action='store_true', help='Process sessions in parallel')
    parser.add_argument('--force', action='store_true', help='Process invalid nes sessions')
    parser.add_argument('--profile', action='store_true', help='Profile the code')
    parser.add_argument('--jenkins', action='store_true', help='run for jenkins')
    parser.add_argument('--run_tester', action='store_true', help='Run tester to get data')
    parser.add_argument('-session_ids_train', metavar='ids', nargs='+', type=int, help='Index of setups in DB',
                        required=False)
    parser.add_argument('-session_ids_test', metavar='ids', nargs='+', type=int, help='Index of setups in DB',
                        required=False)
    parser.add_argument('-nn_fs', metavar='fs', type=int, help='sampling frequency for the nn', required=True)
    parser.add_argument('-window', metavar='window', type=int, help='time window of the signal', required=True)
    parser.add_argument('-acc_name', metavar='FileName', type=str, help='name of the accumulated files', required=True)
    parser.add_argument('-load_path', metavar='Location', type=str, help='Path from which to load file', required=False)
    parser.add_argument('-save_path', metavar='Location', type=str, help='location of processed data and results',
                        required=False)
    parser.add_argument('-model_path', metavar='Location', type=str, help='location of the model', required=False)
    parser.add_argument('-seed', metavar='seed', type=int, help='seed for random split of train and validation data',
                        required=False)
    parser.add_argument('-labels', metavar='Labels', nargs='+', type=str,
                        help='compute the following labels (default is all status)',
                        default=[a for a in ['motion', 'speaking', 'zrr', 'occupancy']])
    parser.add_argument('-benchmark', type=str, choices=['for_test', 'for_test_2', 'for_test_3', 'for_test_4',
                                                         'fmcw_raw', 'fmcw_cpx', 'fmcw_raw_front', 'fmcw_cpx_front',
                                                         'rest_failures_hr', 'low_hr', 'high_hr',
                                                         'covid_belinson_sample', 'covid_belinson_dynamic', 'rest_back',
                                                         'motion_back', 'driving_back', 'regression', 'all'],
                        help='benchmark on which to run version testing\n'
                             ' for_test = sessions in for_test column in session list or for_test benchmark in db\n'
                             ' for_test_2 = sessions in for_test_2 column in session list or for_test_2 benchmark in '
                             'db\n'
                             ' for_test_3 = sessions in for_test_3 column in session list or for_test_3 benchmark in '
                             'db\n'
                             ' for_test_4 = sessions in for_test_4 benchmark in db; cherry, rest, no motion, no '
                             'speaking, no driving, no engine on, with continues reference\n'
                             ' fmcw_raw = fmcw raw sessions\n'
                             ' fmcw_cpx = fmcw complex sessions\n'
                             ' fmcw_raw_front = fmcw raw front sessions\n'
                             ' fmcw_cpx_front = fmcw complex front sessions\n'
                             'rest_failures_hr = for_test_4 sessions that failed (less then 90%)\n'
                             ' low_hr = sessions in low_hr benchmark in db\n'
                             ' high_hr = sessions in high_hr benchmark in db\n'
                             ' covid_belinson_sample = sessions in covid_belinson_sample (sample reference) benchmark '
                             'in db\n'
                             ' covid_belinson_dynamic = sessions in covid_belinson_dynamic (continues reference) '
                             'benchmark in db\n'
                             ' rest_back = cw, has reference, back mount rest sessions with no motion, no engine on, '
                             'no speaking and no driving\n'
                             ' motion_back = cw, has reference, back mount sessions with motion, no engine on, no '
                             'speaking and no driving\n'
                             ' driving_back = cw, has reference, back mount sessions with driving, no speaking\n'
                             ' regression = session marked for regression for this version'
                             ' all = run all benchmarks\n')
    parser.add_argument('--no_reliability', action='store_true', help='Dont take reliability into account')
    parser.add_argument('--plot_ref', action='store_true', help='plot nes-ref per session')
    parser.add_argument('--plot_all', action='store_true', help='plot freq-time, psd plots per session')
    parser.add_argument('-cpp', metavar='Location', type=str, required=False, help='location of linux compiled cpp')
    parser.add_argument('-jenkins_result_dir', metavar='Jenk-result', type=str, help='The directory of jenkins results')
    parser.add_argument('-log_path', metavar='Log-Path', type=str, required=False,
                        help='directory to write log file to')
    parser.add_argument('-result_dir', metavar='Results', type=str, required=False,
                        help='file to which to write results')
    parser.add_argument('-version', metavar='Version', type=str, help='Unique name (e.g. commit ID)', default='HR')
    parser.add_argument('-config_path', metavar='Config', type=str, help='yaml file of configuration parameters',
                        default=None)
    parser.add_argument('-compute', metavar='Compute', nargs='+', type=VitalSignType,
                        choices=[a for a in VitalSignType], help='compute the following vital signs (default is all)\n'
                                                                 ' hr = heart rate\n'
                                                                 ' rr = respiration rate\n'
                                                                 ' ra = respiration amplitude\n'
                                                                 ' ie = ratio between inhale and exhale\n'
                                                                 ' hri = heart rate interval\n'
                                                                 ' hrv_nni_mean = NN intervals mean\n'
                                                                 ' hrv_sdnn = NN intervals standard deviation\n'
                                                                 ' hrv_nni_cv = NN intervals covariance\n'
                                                                 ' hrv_lf_max_power = maximum low frequency power\n'
                                                                 ' hrv_hf_max_power = maximum high frequency power\n'
                                                                 ' hrv_lf_auc = low frequency area of under curve\n'
                                                                 ' hrv_hf_auc = high frequency area of under curve\n'
                                                                 ' stat = status\n',
                        default=[a for a in VitalSignType])
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing sessions')
    parser.add_argument('--silent', action='store_true', help='Display only warnings and errors')
    parser.add_argument('-prep_data_window', metavar='window', type=int, required=False, help='time window of the data')
    parser.add_argument('-start_sec', metavar='time', type=int, default=10, help='signal second of start')
    parser.add_argument('--bins', action='store_true', help='take all the bins from fmcw signal')
    parser.add_argument('-gt_path', metavar='Location', type=str, required=False,
                        help='location of ground_truth data if not in DB')
    parser.add_argument('-gt_file_name', metavar='FileName', type=str, default='',
                        help='file names (after id) in gt_path')
    parser.add_argument('-ref_path', metavar='Location', type=str, required=False,
                        help='location of processed reference')
    parser.add_argument('-label_dict_path', metavar='Location', type=str, required=False,
                        help='location of labels dictionary')
    parser.add_argument('--overwrite_model', action='store_true', help='Overwrite existing model')
    parser.add_argument('-mean_std_name', metavar='FileName', type=str,
                        help='name of the initial mean and std accumulated files', required=False)
    parser.add_argument('-skip_long_setups', metavar='seed', type=int, required=False,
                        help='skip sessions longer then the input in seconds')


    return parser.parse_args()


def test_version(versionn, benchmark=None, sessionss=None):
    if sessionss:
        benchmark = 'selected_sessions'
    elif benchmark == 'regression':
        sessionss = gen_regression_session_ids(versionn)
    else:
        sessionss = gen_sessions_ids(benchmark)  # gather_session_ids

    jenk_dict = get_jenkins_variables(args.result_dir, args.jenkins)
    if args.seed is not None:
        testing_version = jenk_dict['cur_commit_id'] if args.jenkins else versionn + '_' + benchmark + '_seed_' + \
                                                                          str(args.seed)
    else:
        testing_version = jenk_dict['cur_commit_id'] if args.jenkins else versionn + '_' + benchmark
    if args.spot_mode:
        testing_version = testing_version + '_sm'
    if args.no_reliability:
        testing_version = testing_version + '_no_rel'
    output_dir = os.path.join(args.result_dir, testing_version)
    create_dir(output_dir)
    args.log_path = output_dir
    os.chmod(output_dir, 0o777)
    run_tester(args, sessionss, testing_version, args.seed)  # run tester


def setup_list():
    db = DB()
    no_driving = set(db.setup_by_state(state=State.is_engine_on, value=False))
    no_driving2 = set(db.setup_by_state(state=State.is_driving, value=False))
    no_driving3 = set(db.setup_by_state(state=State.is_driving_idle, value=False))
    stationary = set(db.setup_vs_equals(VS.stationary, 1))
    tidal = set(db.setup_by_note(note='Tidal'))
    back = set(db.setup_by_target(target=Target.back))
    front = set(db.setup_by_target(target=Target.front))
    firmware = set(db.setup_fw_greater_than(fw='0.4.7.5'))
    fmcw = set(db.sr_fmcw_setups())
    cpx = set(db.setup_by_vs(vs=VS.cpx))
    raw = set(db.setup_by_vs(vs=VS.raw))
    not_both_cpx_raw = cpx.symmetric_difference(raw)
    valid = set(db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.valid))
    confirmed = set(db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.confirmed))
    recorded_here = set(db.setup_by_project(prj=Project.neteera))
    zrr = set(db.setup_vs_equals(VS.zrr, 1))
    ec = set(db.setup_vs_equals(VS.occupancy, 0))
    motion = set(db.setup_by_state(State.is_motion, value=True))
    sitting = set(db.setup_by_posture(Posture.sitting))
    env_lab = set(db.setup_by_environment(environment=Environment.lab))
    gt_definition_amb = set(db.setup_by_note('Station A protocol'))
    seat = set(db.setup_by_mount(Mount.seat_back))
    inters = fmcw & firmware & stationary & recorded_here & env_lab & back & sitting & seat & no_driving & no_driving2 & \
             no_driving3 & (valid | confirmed) & not_both_cpx_raw & (zrr | ec | motion) - tidal \
             - set([3869, 4121, 3407, 3409, 3410, 3411, 3414, 3415, 4156, 3151, 3156, 3157, 3172, 3434, 3436, 4220,
                    4408, 4381, 3664, 4340, 4178, 3456, 3449, 3165, 4285, 3800, 3282, 3519, 3528, 4160, 4341, 3397,
                    3800, 4285, 4485, 4305, 3866, 3534, 4172, 4572, 4397, 4328])  # - gt_definition_amb
    inters = inters.union(set([4349, 4471, 4342, 3838, 3682, 3684])) & seat
    return inters


if __name__ == "__main__":

    args = get_args()
    if not args.benchmark:
        if args.session_ids_train and args.session_ids_test:
            sessions = args.session_ids_train + args.session_ids_test
        else:
            sessions = setup_list()
    else:
        sessions = None
    if args.run_tester:
        test_version(args.version, args.benchmark, sessions)
        load_path = os.path.join(args.result_dir, args.version + '_selected_sessions')
        if args.seed:
            load_path += '_seed_{}'.format(args.seed)

    else:
        load_path = args.load_path
    if not args.save_path:
        args.save_path = os.path.join(load_path, 'prepared_data')
    if not args.model_path:
        args.model_path = args.save_path
    if args.prep_data_window:
        layer_size = args.nn_fs * args.prep_data_window
    else:
        layer_size = args.nn_fs * args.window
    if not (args.session_ids_train and args.session_ids_test):
        args.session_ids_train, args.session_ids_test = train_test_split(list(setup_list()), test_size=0.1,
                                                                         random_state=args.seed)
    prepare_data_command = PYTHONVERSION + ' ./Tests/NN/PrepareData.py' + ' -nn_fs ' + str(args.nn_fs) + ' -window ' \
                           + str(args.window) + ' -acc_name ' + args.acc_name + ' -load_path ' + load_path + \
                           ' -save_path ' + args.save_path + ' -session_ids_train ' + \
                           ' '.join(list(map(str, map(int, args.session_ids_train)))) + ' -session_ids_test ' + \
                           ' '.join(list(map(str, map(int, args.session_ids_test))))
    if args.prep_data_window:
        prepare_data_command = prepare_data_command.replace('-window ' + str(args.window), '-window ' +
                                                            str(args.prep_data_window))
    if args.start_sec:
        prepare_data_command += ' -start_sec ' + str(args.start_sec)
    if args.bins:
        prepare_data_command += ' --bins'
    if args.gt_path:
        prepare_data_command += ' -gt_path ' + args.gt_path
    if args.gt_file_name:
        prepare_data_command += ' -gt_file_name ' + args.gt_file_name
    if args.ref_path:
        prepare_data_command += ' -ref_path ' + args.ref_path
    if args.mean_std_name:
        prepare_data_command += ' -mean_std_name ' + args.mean_std_name
    print(prepare_data_command)
    subprocess.run(prepare_data_command, shell=True)
    train_command = PYTHONVERSION + ' ./Tests/NN/TrainNN.py' + ' -data_path ' + args.save_path + ' -save_path ' + \
                    args.model_path + ' -layer_size ' + str(layer_size) + ' -window ' + str(args.window)
    if args.prep_data_window:
        train_command = train_command.replace('-window ' + str(args.window), '-window ' + str(args.prep_data_window))
    if args.seed:
        train_command += ' -seed {}'.format(args.seed)
    if args.labels:
        train_command += ' -labels ' + ' '.join(list(map(str, args.labels)))
    if args.label_dict_path:
        train_command += ' -label_dict_path ' + args.label_dict_path
    if args.overwrite_model:
        train_command += ' --overwrite'
    if args.mean_std_name:
        train_command += ' --mean_std'
    print(train_command)
    subprocess.run(train_command, shell=True)
    test_command = PYTHONVERSION + ' ./Tests/NN/TestNN.py' + ' -data_path ' + args.save_path + ' -model_path ' + \
                   args.model_path + ' -fs ' + str(args.nn_fs)
    if args.mean_std_name:
        test_command += ' --mean_std'
    print(test_command)
    subprocess.run(test_command, shell=True)
