import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa  # noqa
import argparse
import numpy.matlib as ml
from sklearn import preprocessing
from scipy.interpolate import interp1d

from pylibneteera.sp_utils import downsample

from Tests.Utils.DBUtils import *


def data_for_setup(save_path, list_sig, list_ref, list_mean, sig, mean_std, refs, idxx, sess_fr, nn_fs, window, bins=None, reff_path=None):
    """ Generate data and reference for each session and add them together
    :param str save_path: location of output reference adjust to the time window
    :param list list_sig: list of data arrays (X) for each setup
    :param list list_ref: list of reference arrays (y) for each setup
    :param array sig: signal of current setup accumulated by Tester.py
    :param array/list refs: reference from the DB or gt_path
    :param int idxx: setup index
    :param int/float sess_fr: sampling frequency of the setup
    :param int nn_fs: sampling frequency required for the prepared data
    :param int window: time window required for the prepared data
    :param bool bins: determines if to use all the bins in fmcw signal
    :param str reff_path: location of prepared reference to use
    :return: list_ref, list_sig - lists of data and reference, containing data and reference of current setup
    :rtype: list
    """
    if len(sig) == 0:
        return list_ref, list_sig
    if reff_path is not None and os.path.exists(os.path.join(reff_path, str(idxx) + '_ref.npy')):
        reff = np.load(os.path.join(reff_path, str(idxx) + '_ref.npy'), allow_pickle=True)
    elif type(refs) == list:
        reff = np.zeros(len(refs[0]))
        for r in range(len(refs)):
            reff[refs[r] == 1] = r + 1
    else:
        reff = refs
    if (len(reff.shape) < 2) or (not reff.shape[1] == window):
        if bins:
            new_ref = -1 * np.ones((min(int(sig.shape[0] / 2), reff.shape[0] - window), window))
        else:
            new_ref = -1 * np.ones((min(sig.shape[0], reff.shape[0] - window), window))
        for s in np.arange(window, new_ref.shape[0] + window):
            new_ref[s - window] = reff[s - window:s]
        reff = new_ref
    np.save((os.path.join(save_path, str(window) + '_sec_ref/' + str(idxx) + '_ref.npy')), reff)
    if not bins and len(sig.shape) == 3:
        sig = sig[:, :, 0]
    if int(sess_fr) > nn_fs:
        if bins:
            net_size_seg = np.zeros((sig.shape[0], nn_fs * int(sig.shape[1]/int(sess_fr)), sig.shape[2]))
            for b in range(sig.shape[2]):
                net_size_seg[:, :, b] = downsample(sig[:, :, b].T, int(sess_fr), nn_fs)[0].T
        else:
            net_size_seg = downsample(sig.T, int(sess_fr), nn_fs)[0].T
    elif int(sess_fr) < nn_fs:
        from_time = np.linspace(0, int(sig.shape[1]/int(sess_fr)), sig.shape[1])
        upsampled_time = np.linspace(0, int(sig.shape[1] / int(sess_fr)), int(sig.shape[1]/int(sess_fr))*nn_fs)
        if bins:
            net_size_seg = np.zeros((sig.shape[0], nn_fs * int(sig.shape[1] / int(sess_fr)), sig.shape[2]))
            for b in range(sig.shape[2]):
                f = interp1d(from_time, sig[:, :, b])
                net_size_seg[:, :, b] = f(upsampled_time)
        else:
            f = interp1d(from_time, sig)
            net_size_seg = f(upsampled_time)
    else:
        net_size_seg = sig
    if not int(sig.shape[1]/int(sess_fr)) == window:
        net_size_seg = net_size_seg[:, :int(net_size_seg.shape[1] * (window / int(sig.shape[1]/int(sess_fr))))]
    if bins:
        net_size_seg = net_size_seg[1::2, :, :]
        if net_size_seg.shape[0] > reff.shape[0]:
            net_size_seg = net_size_seg[:reff.shape[0], :, :]
        list_sig.append(net_size_seg.transpose(2, 0, 1).reshape(-1, net_size_seg.shape[1]))
        list_ref.append(ml.repmat(reff, net_size_seg.shape[2], 1))
    else:
        if net_size_seg.shape[0] > reff.shape[0]:
            net_size_seg = net_size_seg[:reff.shape[0], :]
        if args.mean_std_name:
            if nn_fs > 20:
                tmp = downsample(net_size_seg.T, nn_fs, 20)[0].T
            else:
                tmp = net_size_seg
            first_mean = np.asarray([mean_std[i]['first_mean'] for i in range(net_size_seg.shape[0])])
            first_std = np.asarray([mean_std[i]['first_std'] for i in range(net_size_seg.shape[0])])
            first_diff_mean = np.asarray([mean_std[i]['first_diff_mean'] for i in range(net_size_seg.shape[0])])
            first_diff_std = np.asarray([mean_std[i]['first_diff_std'] for i in range(net_size_seg.shape[0])])
            list_mean.append(np.asarray([abs(net_size_seg).mean(axis=1) / first_mean,
                             net_size_seg.std(axis=1) / first_std,
                             abs(np.diff(tmp)).mean(axis=1) / first_diff_mean,
                             np.diff(tmp).std(axis=1) / first_diff_std]).T)
        list_sig.append(net_size_seg)
        list_ref.append(reff)
    return list_ref, list_sig, list_mean


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-session_ids_test', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=False)
    parser.add_argument('-session_ids_train', metavar='ids', nargs='+', type=int, help='Setup IDs in DB',
                        required=False)
    parser.add_argument('-load_path', metavar='LoadPath', type=str, required=True, help='Path from which to load files')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output data')
    parser.add_argument('-gt_path', metavar='Location', type=str, required=False, 
                        help='location of ground_truth data if not in DB')
    parser.add_argument('-ref_path', metavar='Location', type=str, required=False, 
                        help='location of processed reference')
    parser.add_argument('-acc_name', metavar='FileName', type=str, required=True,
                        help='name of signal accumulator files')
    parser.add_argument('-nn_fs', metavar='fs', type=int, required=True, help='sampling frequency for the NN')
    parser.add_argument('-window', metavar='window', type=int, required=True, help='time window of the data')
    parser.add_argument('-start_sec', metavar='time', type=int, default=10, help='signal second of start')
    parser.add_argument('--bins', action='store_true', help='take all the bins from fmcw signal')
    parser.add_argument('-mean_std_name', metavar='FileName', type=str, required=False,
                        help='name of mean_std accumulator files')
    parser.add_argument('-gt_file_name', metavar='FileName', type=str, default='',
                        help='file names (after id) in gt_path')

    return parser.parse_args()


if __name__ == "__main__":

    """ Generates data for neural network train (TrainNN) and test (TestNN)
    Use Tester.py results with accumulator and generate X, X_processed and y for train and test 
    """
    db = DB()
    args = get_args()
    if not args.save_path:
        args.save_path = os.path.join(args.load_path, 'prepared_data')
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(os.path.join(args.save_path, str(args.window) + '_sec_ref')):
        os.makedirs(os.path.join(args.save_path, str(args.window) + '_sec_ref'))

    train_setups = args.session_ids_train
    test_setups = args.session_ids_test
    np.save(os.path.join(args.save_path, 'train_setups.npy'), train_setups)
    np.save(os.path.join(args.save_path, 'test_setups.npy'), test_setups)
    if not args.gt_path:
        status = ['motion', 'speaking', 'zrr', 'occupancy']
        label_dict = {'rest': 0}
        for n, vss in enumerate(status):
            label_dict[vss] = n + 1
        np.save(os.path.join(args.save_path, 'label_dict.npy'), label_dict)

    for k in [train_setups, test_setups]:
        if k is None:
            continue
        listsig = []
        listref = []
        listmean = []
        for idx in k:
            try:
                print(idx)
                if db.setup_fs(idx) != 500:
                    print('fs not equal 500, skipping')
                    continue
                if db.setup_duration(idx) < 45:
                    print('session shorter than 45 seconds, skipping')
                    continue

                aligned_refs = []
                setup_sig = np.load(os.path.join(args.load_path, str(idx)+args.acc_name+'.npy'), allow_pickle=True)
                if args.gt_path:
                    aligned_refs = np.load(os.path.join(args.gt_path, str(idx)+args.gt_file_name+'.npy'),
                                           allow_pickle=True)
                else:
                    for vs in status:
                        if vs == 'motion':
                            v = 'GT_REST'
                        else:
                            v = 'GT_' + vs.upper()
                        ref_path = db.setup_ref_path(setup=idx, sensor=Sensor[
                            db.sensor_by_vs('rest' if vs == 'motion' else vs)[0].lower()])[0]
                        if vs in ['motion', 'occupancy']:
                            ref = np.logical_not(load_ref(path=ref_path, sensor_type=ref_sensor_type(v), 
                                                          vital_sign_type=VitalSignType('stat')).astype('int'))
                        else:
                            ref = load_ref(path=ref_path, sensor_type=ref_sensor_type(v), 
                                           vital_sign_type=VitalSignType('stat')).astype('int')
                        t_sig = np.arange(len(setup_sig) + args.start_sec)
                        new_t_sig, aligned_ref, timeshift = match_lists_by_ts(t_sig, ref, idx, vs, db)
                        if new_t_sig[0] > 0:
                            setup_sig = setup_sig[new_t_sig[0]:, :]
                        aligned_refs.append(aligned_ref)
    
                fr = db.setup_fs(idx)
                print(fr)
                if args.mean_std_name:
                    mean_std = np.load(os.path.join(args.load_path, str(idx)+args.mean_std_name+'.npy'),
                                       allow_pickle=True)
                else:
                    mean_std = []
                listref, listsig, listmean = data_for_setup(args.save_path, listsig, listref, listmean, setup_sig,
                                                            mean_std, aligned_refs, idx, fr, args.nn_fs, args.window,
                                                            args.bins, args.ref_path)
            except (FileNotFoundError, NameError, IndexError, AssertionError):
                continue
        if k == train_setups:
            name = 'train'
        elif k == test_setups:
            name = 'test'
        np.save(os.path.join(args.save_path, 'X_{}_{}hz'.format(name, args.nn_fs)), np.vstack(listsig))  # displacement
        if args.mean_std_name:
            np.save(os.path.join(args.save_path, 'X_mean_{}'.format(name)), np.vstack(listmean))  # mean and std
        np.save(os.path.join(args.save_path, 'y_{}_{}hz'.format(name, args.nn_fs)), np.vstack(listref))  # gt
        X0 = np.vstack(listsig)
        X = np.zeros((X0.shape[0], 2, args.nn_fs * args.window))
        for i in range(X0.shape[0]):
            #
            X1 = np.hstack(X0[i])
            X[i][0] = preprocessing.scale(X1.real)
            m = [0]
            m.extend(np.diff(X1.real))
            X[i][1] = preprocessing.scale(m)
        np.save(os.path.join(args.save_path, 'X_{}_processed_{}hz'.format(name, args.nn_fs)), X)
