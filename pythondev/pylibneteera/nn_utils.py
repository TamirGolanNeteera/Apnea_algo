# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from keras.models import model_from_json
import logging
import numpy as np
import os
from sklearn import preprocessing

from pylibneteera.sp_utils import resample_sig


def load_model(model_path):
    model = None
    if os.path.isfile(os.path.join(os.getcwd(), 'NN', model_path['json'])) and os.path.isfile(
            os.path.join(os.getcwd(), 'NN', model_path['hdf5'])):
        logging.getLogger('vsms').debug('Loading speaking motion model and weights...\n{}\n{}\n'.format(
            os.path.join(os.getcwd(), 'NN', model_path['json']),
            os.path.join(os.getcwd(), 'NN', model_path['hdf5'])))
        with open(os.path.join(os.getcwd(), 'NN', model_path['json']), 'r') as m:
            model = model_from_json(m.read())
        model.load_weights(os.path.join(os.getcwd(), 'NN', model_path['hdf5']))
        logging.getLogger('vsms').debug('Speaking motion model and weights loaded\n')
    else:
        logging.getLogger('vsms').warning('Speaking motion model and / or weights not found\n{}\n{}\n'.format(
            os.path.join(os.getcwd(), 'NN', model_path['json']),
            os.path.join(os.getcwd(), 'NN', model_path['hdf5'])))
    return model


def prep_for_pred(dat, nn_fs, **kwargs):
    first_mean_std = kwargs.get('mean_std', None)
    if len(dat.shape) == 2:
        dat = dat[:, 0]
    net_size_seg = resample_sig(dat, dat.fs, nn_fs)
    norm_sig = preprocessing.scale(net_size_seg)
    diff_data = [0]
    diff_data.extend(np.diff(net_size_seg))
    norm_diff = preprocessing.scale(diff_data)
    if first_mean_std:
        means = np.asarray([abs(net_size_seg).mean() / first_mean_std['first_mean'],
                            net_size_seg.std() / first_mean_std['first_std'],
                            abs(np.diff(net_size_seg)).mean() / first_mean_std['first_diff_mean'],  # TODO: add downsample to 20 for the diff
                            np.diff(net_size_seg).std() / first_mean_std['first_diff_std']]).T  # TODO: add downsample to 20 for the diff
        return norm_sig, norm_diff, means
    else:
        return norm_sig, norm_diff


def calc_means(phase, nn_fs):
    sig = resample_sig(phase, phase.fs, nn_fs)  # TODO: add downsample to 20 for the diff
    return {'first_mean': abs(sig).mean(),
            'first_std': sig.std(),
            'first_diff_mean': abs(np.diff(sig)).mean(),
            'first_diff_std': np.diff(sig).std()}

