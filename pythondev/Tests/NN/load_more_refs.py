# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())
import numpy as np
import argparse

from Tests.vsms_db_api import *

db = DB()


def get_args() -> argparse.Namespace:
 """ Argument parser

 :return: parsed arguments of the types listed within the function
 :rtype: argparse.Namespace
 """
 parser = argparse.ArgumentParser(description='Process some integers.')

 return parser.parse_args()


if __name__ == '__main__':
    ecg1 = np.load('/Neteera/DATA/2022/6/29/106422/109921/REFERENCE/RESPIRONICS_ALICE6/ECG_I_109921_007.1-NC-F2-R1-007_1656439381.npy', allow_pickle=True)
    print("...")