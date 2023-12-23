# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))   # noqa

from Offline import load_and_run_algo
from Tests.Utils.DBUtils import get_not_invalid_setups

from Tests.PBSCOntroller import setups_done, JobsController
from Tests.vsms_db_api import *
from Tests.OfflineOutputHandler import OfflineOutputHandler
from Tests.Utils.TestsUtils import run_cmd, intersect
from Tests.VSParser import *
from Tests.Utils.PathUtils import create_dir
from Tests.Utils.TesterUtils import setup_logger

from pylibneteera.datatypes import VitalSignType

from time import time
import argparse
import logging
import copy


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = vs_parser(description='Neteera vital signs tracker on offline data files')
    parser.add_argument('-setups', '-session_ids', '-setup_ids', '-setup', nargs='+', type=int, help='setup list',
                        default=[5755])
    parser.add_argument('--parallel', action='store_true', help='Process setups in parallel')
    parser.add_argument('--profile', action='store_true', help='Profile the code')
    parser.add_argument('-start_time', type=int, help='run the tester from X seconds from the start', default=0)
    parser.add_argument('-end_time', type=int, help='end the tester at X seconds from the start', default=0)
    return post_parsing(parser.parse_args())


def remove_finished_setups(ids, out_dir):
    results = setups_done(ids, out_dir)
    return list(set(ids) - set(results))


def remove_invalid_setups(ids, database):
    database.update_mysql_db(ids[0])
    if database.mysql_db == 'neteera_cloud_mirror':
        valid = ids
    else:
        valid = get_not_invalid_setups(database)
    return intersect([valid, ids])


def remove_long_setups(max_duration, ids, database):
    return [idx for idx in ids if database.setup_duration(idx) < max_duration]


def remove_setups_outside_the_db(ids, database):
    new_list = []
    for idx in ids:
        if os.path.exists(database.setup_ref_path(idx, Sensor.nes)[0]):
            new_list.append(idx)
        else:
            print(f'setup {idx} was not uploaded to the db, probably recorded in semi-online mode.'
                  f'\nIf it is more then one day old than contact the operator')
    return new_list


def filter_out_setups(argss, db, out_dir):
    ids = argss.setups
    if not argss.force:
        ids = remove_invalid_setups(ids, db)
    if argss.skip_long_setups is not None and argss.skip_long_setups > 0:
        ids = remove_long_setups(argss.skip_long_setups, ids, db)
    if ids and not argss.overwrite:
        ids = remove_finished_setups(ids, out_dir)
    db.update_mysql_db(ids[0])
    ids = remove_setups_outside_the_db(ids, db)
    return sorted(ids)


def validate_inputs(bargs):
    if os.getcwd().endswith('Tests') or os.getcwd().endswith('Tests/'):
        os.chdir('..')
        print(f'working dir changed to {os.getcwd()}')
    assert not (bargs.parallel and bargs.profile), 'Cannot profile a parallel run\r\n'
    if VitalSignType.rr not in bargs.compute:
        bargs.compute += [VitalSignType.rr]
    if (VitalSignType.bbi in bargs.compute or VitalSignType.stat in bargs.compute) \
            and VitalSignType.hr not in bargs.compute:
        bargs.compute += [VitalSignType.hr]
    return bargs


def tester_main(args: argparse.Namespace):
    args = validate_inputs(args)
    output_dir = os.path.join(args.result_dir, args.version)
    create_dir(output_dir)

    setup_logger(os.path.join(output_dir, 'logs'), args.silent, args.setups)

    db = DB()

    ids = filter_out_setups(args, db, output_dir)
    total = len(ids)
    assert total, 'no relevant setups were found, try changing a db or using --force'
    logging.getLogger('vsms').warning(f'Evaluating: {ids}\r\n')
    start_time = time()
    if args.parallel and platform.system() == 'Linux':
        control = JobsController(args, output_dir)
        control.run_parallel(ids, total)
    else:
        result_handler = None
        for idw in sorted(ids):
            copy_args = copy.deepcopy(args)
            if idw not in db.setup_by_vs(VS.hri, Sensor.epm_10m):
                logging.getLogger('vsms').warning(
                    f'setup {idw} has no bbi ref, running in spot mode only hr, rr, bio-id')
                copy_args.compute = list(set(args.compute) - {'identity'})
            if not args.silent:
                result_handler = OfflineOutputHandler(idx=idw, db=db, vital_signs=args.compute)
            load_and_run_algo(args, idw, output_dir, result_handler, db)
    logging.getLogger('vsms').warning('Total algorithm runtime: {} Seconds'.format(round(time() - start_time)))
    if len(ids) == 1:
        for vs in args.plot_vs:
            result_path = os.path.join(output_dir, 'dynamic',  f'{ids[0]}_{vs}.') + ('data' if vs == 'stat' else 'npy')
            run_cmd(f' ./Tests/Plots/PlotResults.py -result_dir {result_path} --force')


if __name__ == "__main__":
    tester_main(get_args())
