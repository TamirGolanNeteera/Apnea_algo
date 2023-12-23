# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa

from Tests.vsms_db_api import *

import pandas as pd
import os
import re
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-data_location', metavar='LoadPath', type=str, required=True,
                        help='data_location')
    parser.add_argument('-dist_error', type=int, required=False, default='150')
    parser.add_argument('--csv', action='store_true', help='vital sign tracker results from csv files instead of npy \
     files')
    return parser.parse_args()


def get_session_list_py(folder):
    return [re.findall(r'\d+', file)[0] for file in os.listdir(folder)
            if re.fullmatch(f"[0-9]+_estimated_range.npy", file)]


def get_session_list_csv(folder):
    return [re.findall(r'\d+', file)[0] for file in os.listdir(folder)
            if re.fullmatch(f"[0-9]+.csv", file)]


def main_range_evaluator(folder, dist_error, csv):
    db = DB()
    error_dict = {}
    result_df = pd.DataFrame(columns=['Session', 'distance_estimated', 'distance_DB', 'bin_estimated', 'db_bin',
                                      'gain_DB', 'db_adc_sample_num', 'db_bandwidth', 'db_maxBinsToStream',
                                      'db_binToStreamOffset', 'error_under_{}_mm'.format(dist_error)])
    if csv:
        session_lists = get_session_list_csv(folder)
    else:
        session_lists = get_session_list_py(folder)
    session_lists = sorted(session_lists)
    for i, ses in enumerate(session_lists):
        db_dist = int(db.setup_distance(setup=ses))
        if db_dist >= 0:
            if csv:
                estimation_csv = pd.read_csv(os.path.join(folder, f'{ses}.csv')).loc[0]
                estimation_dist = estimation_csv['range_selected']
                estimation_bin = estimation_csv['bin_selected']
            else:
                estimation_npy = np.load(os.path.join(folder, f'{ses}_estimated_range.npy'), allow_pickle=True)[0]
                estimation_dist = estimation_npy[0]
                estimation_bin = estimation_npy[1]
            radar_cfg_db = db.setup_radar_config(setup=ses)
            db_bin = db.setup_bin(setup=ses)
            db_bin = db_bin.get('bin')
            db_gain = [radar_cfg_db['opAmplControl_pgaI1'], radar_cfg_db['opAmplControl_pgaI2']]
            db_adc_sample_num = radar_cfg_db['basebandConfig_ADC_samplesNum']
            db_bandwidth = radar_cfg_db['PLLConfig_bandwidth']
            db_max_bins_to_stream = radar_cfg_db['systemConfig_maxBinsToStream']
            db_bin_to_stream_offset = radar_cfg_db['systemConfig_binToStreamOffset']
            error = 1 if abs(db_dist - estimation_dist) <= dist_error else 0
            error_dict[db_dist] = error_dict.get(db_dist, []) + [error]

            result_df.loc[i] = [ses, estimation_dist, db_dist, estimation_bin, db_bin, db_gain,
                                db_adc_sample_num, db_bandwidth, db_max_bins_to_stream, db_bin_to_stream_offset, error]

    stat_df = \
        pd.DataFrame(columns=['Distance', 'amount_of_sessions', f'total_score_under_{dist_error}_mm_error %'])
    for i, k in enumerate(error_dict.keys()):
        stat_df.loc[i] = [k, len(error_dict[k]), round(sum(error_dict[k]) / len(error_dict[k]) * 100)]

    amount_for_weighted = np.array(stat_df['amount_of_sessions'])
    errors_for_weighted = np.array(stat_df[f'total_score_under_{dist_error}_mm_error %'])
    weighted_error = np.average(errors_for_weighted, weights=amount_for_weighted)
    print(stat_df)
    weighted_error_df = pd.DataFrame(columns=['Distance',
                                              'amount_of_sessions',
                                              f'total_score_under_{dist_error}_mm_error %'])
    weighted_error_df.loc[0] = ['Weighted average', sum(amount_for_weighted), weighted_error]
    print(weighted_error_df)
    save_path = os.path.join(folder, f'estimate_results_{dist_error}mm_error.csv')
    pd.concat([stat_df, weighted_error_df, result_df]).to_csv(save_path, index=False)


if __name__ == '__main__':
    args = get_args()
    main_range_evaluator(args.data_location, args.dist_error, args.csv)
