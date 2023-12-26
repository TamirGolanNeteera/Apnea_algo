# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
""" compares Bit Exact between two Result CSV files that are under the same name
 inputs: folder containing CSV dir, epsilon """
import pandas as pd
import os
import re
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-data_location', metavar='csv_list', type=str, nargs='+', required=True,
                        help='list of data_locations')
    parser.add_argument('-epsilon', type=float, required=False, default='0.01')

    return parser.parse_args()


def get_setup_list(folder):
    return [re.findall(r'\d+', file)[0] for file in os.listdir(folder) if re.fullmatch(fr"[0-9]+\.csv", file)]


def bit_exact_columns(col1, col2, epsilon):
    return (abs(col1 - col2) <= epsilon).eq(True).all()


def compare(dirs, epsilon):
    result_df = None
    setup_lists = [set(get_setup_list(folder)) for folder in dirs]
    common_setups = set.intersection(*setup_lists)
    common_setups = sorted(common_setups)
    for i, sess in enumerate(common_setups):
        df1 = pd.read_csv(os.path.join(dirs[0], f'{sess}.csv'))
        df2 = pd.read_csv(os.path.join(dirs[1], f'{sess}.csv'))
        try:   # continuous
            if len(df1['hr']) != len(df2['hr']):
                print('setup {} has different length'.format(sess))
                continue
            if result_df is None:
                result_df = pd.DataFrame(columns=['setup', 'Epsilon', 'hr_bit_exact', 'rr_bit_exact'])
            hr_bit_exact = bit_exact_columns(df1['hr'], df2['hr'], epsilon)
            rr_bit_exact = bit_exact_columns(df1['rr'], df2['rr'], epsilon)
            result_df.loc[i] = [sess, epsilon, hr_bit_exact, rr_bit_exact]

        except KeyError:      # spot
            hrMed_bit_exact = bit_exact_columns(df1['hrMed'], df2['hrMed'], epsilon)
            hrMin_bit_exact = bit_exact_columns(df1['hrMin'], df2['hrMin'], epsilon)
            hrMax_bit_exact = bit_exact_columns(df1['hrMax'], df2['hrMax'], epsilon)
            hrConf_bit_exact = bit_exact_columns(df1['hrConf'], df2['hrConf'], epsilon)
            rrMed_bit_exact = bit_exact_columns(df1['rrMed'], df2['rrMed'], epsilon)
            rrMin_bit_exact = bit_exact_columns(df1['rrMin'], df2['rrMin'], epsilon)
            rrMax_bit_exact = bit_exact_columns(df1['rrMax'], df2['rrMax'], epsilon)
            rrConf_bit_exact = bit_exact_columns(df1['rrConf'], df2['rrConf'], epsilon)
            if result_df is None:
                result_df = pd.DataFrame(columns=['setup', 'Epsilon',
                                                  'hrMed_bit_exact', 'hrMin_bit_exact',
                                                  'hrMax_bit_exact', 'hrConf_bit_exact',
                                                  'rrMed_bit_exact', 'rrMin_bit_exact',
                                                  'rrMax_bit_exact', 'rrConf_bit_exact'])
            result_df.loc[i] = [sess, epsilon,
                                hrMed_bit_exact, hrMin_bit_exact, hrMax_bit_exact, hrConf_bit_exact,
                                rrMed_bit_exact, rrMin_bit_exact, rrMax_bit_exact, rrConf_bit_exact]
    result_df.to_csv(os.path.join(dirs[0], 'bit-exact_epsilon-{}_results.csv'.format(epsilon)), index=False)
    print(result_df)
    return


if __name__ == '__main__':
    args = get_args()
    dirs = [args.data_location[0], args.data_location[1]]
    epsilon = args.epsilon
    compare(dirs, epsilon)
