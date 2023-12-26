import argparse
import keras
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-data_path', metavar='LoadPath', type=str, required=True, help='Path from which to load files')
    parser.add_argument('-model_path', metavar='Location', type=str, required=False, help='location of the model')
    parser.add_argument('-save_path', metavar='Location', type=str, required=False, help='location of output results')
    parser.add_argument('-fs', metavar='fs', type=int, required=True, help='Sampling frequency of the data')
    parser.add_argument('--mean_std', action='store_true', help='test on nn contains also mean and std data')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = args.data_path

    if args.save_path:
        path_to_pred_out_dir = args.save_path
    else:
        path_to_pred_out_dir = args.data_path
    if not os.path.isdir(path_to_pred_out_dir):
        os.makedirs(path_to_pred_out_dir)
    path_to_ref_data = os.path.join(args.data_path, 'y_test_' + str(args.fs) + 'hz.npy')
    path_to_sig_data = os.path.join(args.data_path, 'X_test_processed_' + str(args.fs) + 'hz.npy')
    X = np.load(path_to_sig_data, allow_pickle=True)
    y = np.load(path_to_ref_data, allow_pickle=True)
    if args.mean_std:
        path_to_mean_sig_data = os.path.join(args.data_path, 'X_mean_test.npy')
        X_mean = np.load(path_to_mean_sig_data, allow_pickle=True)

    window = int(X.shape[2]/args.fs)
    label_dict = np.load(os.path.join(args.data_path, 'label_dict.npy'), allow_pickle=True).item()
    new_dict = np.load(os.path.join(model_path, 'new_label_dict.npy'), allow_pickle=True).item()

    new_X = []
    new_mean_X = []
    new_y = []

    for k, val in new_dict.items():
        if (val == 0) | (k == 'zrr'):
            new_X.append(X[np.count_nonzero(y == label_dict[k], axis=1) == window])
            if args.mean_std:
                new_mean_X.append(X_mean[np.count_nonzero(y == label_dict[k], axis=1) == window])
        else:
            new_X.append(X[np.count_nonzero(y == label_dict[k], axis=1) >= 2])
            if args.mean_std:
                new_mean_X.append(X_mean[np.count_nonzero(y == label_dict[k], axis=1) >= 2])
        new_y.append(val * np.ones(new_X[val].shape[0]))
    X = np.vstack(new_X)
    y = np.hstack(new_y)
    if args.mean_std:
        X_mean = np.vstack(new_mean_X)
    assert os.path.isfile(model_path + '/checkpoints/model.hdf5'), 'model not found in {}'.format(model_path)
    print('loading checkpoint and predicting...')
    json_file = open(model_path + '/checkpoints/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights(model_path + '/checkpoints/model.hdf5')
    if args.mean_std:
        preds = model.predict([np.expand_dims(np.stack((X[:, 0, :], X[:, 1, :]), axis=2), axis=3),
                               np.expand_dims(X_mean, axis=2)])
    else:
        preds = model.predict(np.expand_dims(np.stack((X[:, 0, :], X[:, 1, :]), axis=2), axis=3))
    binary = False
    if preds.shape[1] == 1:
        binary = True
    if binary:
        if not os.path.exists(os.path.join(path_to_pred_out_dir, 'fn')):
            os.mkdir(os.path.join(path_to_pred_out_dir, 'fn'))
            os.mkdir(os.path.join(path_to_pred_out_dir, 'fp'))
            os.mkdir(os.path.join(path_to_pred_out_dir, 'tn'))
            os.mkdir(os.path.join(path_to_pred_out_dir, 'tp'))
        fn_counts = 0
        fp_counts = 0
        tn_counts = 0
        tp_counts = 0
        tot_counts = 0
        tot_true_counts = 0
        tot_false_counts = 0

        thresh = 0.5

        for itr, p in enumerate(preds):
            tot_counts += 1
            p = int(p > thresh)
            y[itr] = int(y[itr])
            if p == 1 and y[itr] == 0:  # FP
                plt.subplot(211)
                plt.plot(X[itr, 0])
                plt.title('false positive, gt label = no noise')
                plt.subplot(212)
                plt.plot(X[itr, 1])
                plt.savefig(os.path.join(path_to_pred_out_dir, 'fp') + '/{}.png'.format(itr))
                plt.clf()
                fp_counts += 1
                tot_false_counts += 1
            if p == 0 and y[itr] == 1:  # FN
                plt.subplot(211)
                plt.plot(X[itr, 0])
                plt.title('false negative, gt label = noise')
                plt.subplot(212)
                plt.plot(X[itr, 1])
                plt.savefig(os.path.join(path_to_pred_out_dir, 'fn') + '/{}.png'.format(itr))
                plt.clf()
                fn_counts += 1
                tot_true_counts += 1
            if p == 1 and y[itr] == 1:  # TP
                plt.subplot(211)
                plt.plot(X[itr, 0])
                plt.title('true positive, gt label = noise')
                plt.subplot(212)
                plt.plot(X[itr, 1])
                plt.savefig(os.path.join(path_to_pred_out_dir, 'tp') + '/{}.png'.format(itr))
                plt.clf()
                tp_counts += 1
                tot_true_counts += 1
            if p == 0 and y[itr] == 0:  # TN
                plt.subplot(211)
                plt.plot(X[itr, 0])
                plt.title('true negative, gt label = no noise')
                plt.subplot(212)
                plt.plot(X[itr, 1])
                plt.savefig(os.path.join(path_to_pred_out_dir, 'tn') + '/{}.png'.format(itr))
                plt.clf()
                tn_counts += 1
                tot_false_counts += 1
        
        all_preds = [int(pp[0] > thresh) for pp in preds]
        plt.plot(y, linewidth=3, label='ground truth')
        plt.plot(all_preds, alpha=0.7, label='NN binary prediction')
        plt.xlabel('samples [#]')
        plt.ylabel('noise [1] / no noise [0]')
        plt.legend(fontsize='small', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.savefig(os.path.join(path_to_pred_out_dir, 'overall.png'))

        fnr = fn_counts / (fn_counts + tp_counts)
        fpr = fp_counts / (fp_counts + tn_counts)
        fdr = fp_counts / (fp_counts + tp_counts)
        forr = fn_counts / (fn_counts + tn_counts)

        print('tot_counts: {} sec\n tot_true_count: {} sec\n tot false count: {} sec\n '
              'total false positive (false alarms): {} sec\n total false negative (miss): {} sec\n '
              'total true positive: {} sec\n total true negative: {} sec\n false negative rate / miss rate: {}%\n '
              'false positive rate / fall-out:  {}%\n false discovery rate {}%\n false omission rate {}%\n'.
              format(tot_counts, tot_true_counts, tot_false_counts, fp_counts, fn_counts, tp_counts, tn_counts, 100*fnr,
                     100*fpr, 100*fdr, 100*forr))

    else:

        preds = np.argmax(preds, axis=1)
        conf_mat = confusion_matrix(y, preds)
        conf_mat_norm = confusion_matrix(y, preds, normalize='true')

        print(conf_mat)
        print(conf_mat_norm)

        np.save(os.path.join(path_to_pred_out_dir, 'gt.npy'), y)
        np.save(os.path.join(path_to_pred_out_dir, 'preds.npy'), preds)
        np.save(os.path.join(path_to_pred_out_dir, 'confusion_matrix.npy'), conf_mat)
