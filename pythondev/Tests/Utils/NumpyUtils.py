import numpy as np


def weighted_average(values: iter, weights: iter) -> float:
    try:
        return np.average(values, weights=weights)
    except ZeroDivisionError:
        return np.nan

