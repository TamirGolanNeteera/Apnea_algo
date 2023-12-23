# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import os
import argparse
from argparse import RawTextHelpFormatter
from datetime import date

from Tests.Utils.StringUtils import join
from Tests.Plots.PlotResults import VS_SUPPORTED

from pylibneteera.datatypes import VitalSignType

VITAL_SIGNS_SPOT = ['hr', 'rr', 'bbi', 'hrv_spotsdnn', 'hrv_spotrmssd', 'identity']
VITAL_SIGNS_DYNAMIC = ['hr', 'rr', 'ra', 'ie', 'stat', 'intra_breath', 'posture']


def vs_parser(description: str) -> argparse.ArgumentParser:
    """ A basic parser for command-line running of the algorithm

    :param str description: a description for the parser
    :return: a parser with universally-used arguments for the project
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-version', metavar='Version', type=str, help='Unique name (e.g. commit ID)')
    parser.add_argument('--spot_mode', action='store_true', help='Run tester in spot mode')
    parser.add_argument('-compute', metavar='Compute', nargs='+', type=VitalSignType,
                        choices=list(VitalSignType), help='compute the following vital signs (default is all)\n')
    # log_path not in use
    parser.add_argument('-log_path', metavar='Log-Path', type=str, required=False, help='folder to write log file')
    parser.add_argument('-result_dir', '-save_path', metavar='Results', type=str, required=False,
                        default=os.path.split(os.getcwd())[0], help='file to which to write results')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing sessions')
    parser.add_argument('--silent', action='store_true', help='Display only warnings and errors')
    parser.add_argument('-skip_long_setups', '-skip_long_sessions', type=int, required=False,
                        help='skip sessions longer than given seconds')
    parser.add_argument('-spot_time_from_end', type=int, default=60,
                        help='Run spot mode median on last given seconds.')
    parser.add_argument('--force', action='store_true', help='Process invalid nes sessions')
    parser.add_argument('-classifier', type=str, default='nn', help='Type of classifier, either nn (default) or linear')
    parser.add_argument('-plot_vs', type=str, nargs='+', help='show plot in the end of the session',
                        choices=VS_SUPPORTED, default=[])
    parser.add_argument('--plot_acc', action='store_true', help='save plots of accumulated values')
    parser.add_argument('-seed', help='for backwards compatibility')

    return parser


def post_parsing(args):
    if not args.compute:
        args.compute = VITAL_SIGNS_SPOT if args.spot_mode else VITAL_SIGNS_DYNAMIC
        args.compute = [VitalSignType[vs] for vs in args.compute]
    if not args.version:
        args.version = join(args.compute, sep='_')
        args.version += date.today().strftime("_%d_%m_%Y")
    return args
