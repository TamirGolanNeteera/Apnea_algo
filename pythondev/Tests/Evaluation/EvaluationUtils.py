
import numpy as np
from typing import List


def under_thresh(diffs: np.ndarray, thresh: int) -> List[bool]:
    """ Compare Neteera output to ground truth and return
     mask True when differences are under this percent of ground truth, false otherwise.

    :param np.ndarray diffs: Absolute difference between Neteera output and ground truth
    :param int thresh: thresh to calculate
    :return: A mask of True when differences are under this percent of ground truth, false otherwise
    :rtype: List[bool]
    """
    return diffs <= thresh


def under_percent(diffs: np.ndarray, ground_truth: List, percent) -> List[bool]:
    """ Compare Neteera output to ground truth and return
     mask True when differences are under this percent of ground truth, false otherwise.

    :param np.ndarray diffs: Absolute difference between Neteera output and ground truth
    :param np.ndarray ground_truth: Ground truth
    :param percent: Percent to calculate
    :return: A mask of True when differences are under this percent of ground truth, false otherwise
    :rtype: List[bool]
    """
    fraction = percent / 100
    return diffs <= ground_truth * fraction


def under_percent_or_thresh(diffs: np.ndarray, ground_truth: List, thresh: int, percent: int) -> List[bool]:
    """ Compare Neteera output to ground truth
    return True when differences are under this percent or thresh of ground truth, false otherwise.

    :param np.ndarray diffs: Absolute difference between Neteera output and ground truth
    :param np.ndarray ground_truth: Ground truth
    :param int percent: Percent to calculate
    :param int thresh: thresh to calculate
    :rtype: List[bool]
    """
    return np.logical_or(under_thresh(diffs, thresh), under_percent(diffs, ground_truth, percent))
