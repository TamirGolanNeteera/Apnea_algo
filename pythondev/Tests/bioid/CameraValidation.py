# A scripts that outputs jpg frames from the camera from each setup given, used to validate identity lables

from Tests.Utils.DBUtils import find_back_setup_same_session
from Tests.Utils.PathUtils import create_dir
from Tests.vsms_db_api import *

import argparse
import cv2
import os


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-setup_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB')
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save images',
                        default=os.path.split(os.getcwd())[0])
    return parser.parse_args()


if __name__ == '__main__':
    db = DB()
    args = get_args()
    create_dir(args.save_path)
    setup_ids = {find_back_setup_same_session(x, db) for x in args.setup_ids}

    for setup in setup_ids:
        try:
            path = db.setup_ref_path(setup, Sensor.cam)[0]
            cap = cv2.VideoCapture(path)
            for i in range(3):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(args.save_path, f'{setup}_{db.setup_subject(setup)}.jpg'), gray)
            cv2.destroyAllWindows()

        except IndexError:
            print(f'no camera in setup {setup}')

