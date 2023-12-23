# Utils for numpy arrays, pandas series and dataframes
from Tests.Constants import CATEGORIES
from Tests.Utils.TestsUtils import intersect

from typing import Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nan_first(series):
    if len(series):
        return series.iloc[0]
    else:
        return np.nan


def add_row_to_df(df, index, row):
    if df is None:
        df = pd.DataFrame({index: row}).transpose()
    else:
        if isinstance(index, Tuple) and df.index[-1][0] != index[0]:  # add space between categories
            df = add_tilda(df)
        for metric, value in row.items():
            df.loc[index, metric] = value
    return df


def add_tilda(df):
    if df is None:
        return
    df.loc[(df.index[-1][0], '~~~~~~~~~'), :] = [' '] * df.shape[1]
    return df


def get_common_rows(data_frames: dict):
    setup_nums = [list(dfr.index) for dfr in data_frames.values()]
    common_setup_nums = intersect(setup_nums)
    return {key: dfr.loc[dfr.index.isin(common_setup_nums)] for key, dfr in data_frames.items()}


def get_categories(column_name, values):
    if column_name in CATEGORIES and len(set(values) - set(CATEGORIES[column_name])) == 0:
        return CATEGORIES[column_name]
    else:
        return None


def pd_str_plot(df: Union[pd.DataFrame, pd.Series]):
    df_cat = pd.DataFrame(df)
    if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
        df_cat = df_cat.reset_index(level=[0]).drop('level_0', axis=1)
    if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        df.index = df.index.time
    vs_categories = dict()
    for col_name, col_value in df_cat.iteritems():
        try:
            categorical = pd.Categorical(col_value, categories=get_categories(col_name, col_value))
            vs_categories[col_name] = [x.split()[0] for x in categorical.categories]
            df_cat[col_name] = categorical.codes
        except AttributeError:
            pass    # integer value, not string
    axes = df_cat.plot(style='.', subplots=True)
    for ax, vs in zip(axes, df_cat.columns):
        if vs in vs_categories:
            ax.set_yticks(list(range(len(vs_categories[vs]))))
            ax.set_yticklabels(list(vs_categories[vs]))
        ax.grid()
    return axes


def find_first_entry_of_multi_index(multi_index):
    for i, index in enumerate(multi_index):
        if len(index[1]):
            return i


def get_gap_from_time_series(series):
    delta = series.index[1] - series.index[0]
    return delta.total_seconds()


def is_time_series(obj):
    return isinstance(obj, pd.Series) and isinstance(obj.index, pd.core.indexes.datetimes.DatetimeIndex)
