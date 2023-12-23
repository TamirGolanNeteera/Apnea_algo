# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

import numpy as np


def normal_round(n: float) -> int:
    """ python's round behaves strangely for X.5; use this implementation for c++ compatibility

        :param float n: number to be rounded
        :return: rounded value fo n
        :rtype: int
    """
    if n - np.floor(n) < 0.5:
        return int(np.floor(n))
    return int(np.ceil(n))


def np_normal_round(n: float, decimals: int = 0):
    """ python's numpy round behaves strangely for X.5; use this implementation for c++ compatibility
        np.round(1.5555,3) = 1.556
        np.round(1.5565,3) = 1.556
        np_normal_round(1.5555,3) = 1.556
        np_normal_round(1.5565,3) = 1.557
        :param float n: number to be rounded
        :param int decimals: number of decimal places to round
        :return: rounded value fo n
    """
    if isinstance(n, np.ndarray):
        shape = n.shape
        flattened = n.flatten()
        rounded = np.zeros(flattened.shape)
        for elm_i, elm in enumerate(flattened):
            rounded[elm_i] = normal_round(elm * (10 ** decimals)) / (10 ** decimals)
        return rounded.reshape(shape)
    else:
        return normal_round(n * (10 ** decimals)) / (10 ** decimals)


def max_min(x):
    return np.max(x) - np.min(x)
