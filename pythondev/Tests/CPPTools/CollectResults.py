# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from Tests.Utils.PathUtils import create_dir
from Tests.vsms_db_api import DB, Sensor

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa
import argparse
import os
from shutil import copyfile


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-setups', '-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB',
                        required=True)
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to copy csv to', required=True)
    return parser.parse_args()


def copy_single_setup(path, setup_id, db):
    nes_res_paths = db.setup_ref_path(setup, Sensor.nes_res)
    if nes_res_paths:
        for file in db.setup_ref_path(setup_id, Sensor.nes_res):
            base_name = os.path.basename(file)
            if 'sessionSummary' in base_name:
                copyfile(file, os.path.join(path, 'spot', f'{setup_id}.csv'))
            elif 'results' in base_name:
                copyfile(file, os.path.join(path, 'dynamic', f'{setup_id}.csv'))
            elif 'distance' in base_name:
                copyfile(file, os.path.join(path, 'distance', f'{setup_id}.csv'))
    elif db.mysql_db == 'neteera_cloud_mirror':
        dynamic_results = db.setup_ref_path(setup, Sensor.nes, search='_VS.csv')[0]
        copyfile(dynamic_results, os.path.join(path, 'dynamic', f'{setup_id}.csv'))
        spot_results = db.setup_ref_path(setup, Sensor.nes, search='_spot.csv')[0]
        copyfile(spot_results, os.path.join(path, 'spot', f'{setup_id}.csv'))


if __name__ == '__main__':
    args = get_args()
    db = DB()
    db.update_mysql_db(args.setups[0])
    folder = args.save_path
    if folder.endswith('dynamic') or folder.endswith('dynamic' + os.sep):
        folder = os.path.dirname(folder)
    create_dir(folder)
    modes = ['spot', 'dynamic', 'distance']
    for mode in modes:
        if not create_dir(os.path.join(folder, mode)):
            print(f'{os.path.join(folder, mode)} exists, exiting')
            exit()

    for setup in args.setups:
        copy_single_setup(folder, setup, db)

    os.system(f'chmod 777 -R {folder}')
