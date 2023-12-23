import argparse
import datetime
import pandas as pd
from sklearn.preprocessing import scale
import random
from Tests.NN.reiniers_create_data import *
from Tests.vsms_db_api import *
from Tests.Utils.DBUtils import calculate_delay
from Tests.Utils.LoadingAPI import load_nes
from pylibneteera.sp_utils import resample_sig
import matplotlib.pyplot as plt

def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=False)
    parser.add_argument('-fs', metavar='seed', type=int, required=False, help='New sampling rate')
    parser.add_argument('-nsec', metavar='seed', type=int, required=False, help='Signal length in seconds')
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save data', required=True)
    parser.add_argument('-target', metavar='Target', type=str, help='back or front', required=True)
    parser.add_argument('--clean_train', action='store_true', help='clean high variance sessions from training set')
    return parser.parse_args()

def zero_crossing(x: iter) -> int:

    """number of times the signal crosses the zero"""

    return ((x[:-1] * x[1:]) < 0).sum()




def setuplist():
    db = DB()
    no_driving = set(db.setup_by_state(state=State.is_engine_on, value=False))
    no_driving2 = set(db.setup_by_state(state=State.is_driving, value=False))
    no_driving3 = set(db.setup_by_state(state=State.is_driving_idle, value=False))
    back = set(db.setup_by_target(target=Target.back))
    front = set(db.setup_by_target(target=Target.front))
    fmcw = set(db.sr_fmcw_setups())
    cpx = set(db.setup_by_vs(vs=VS.cpx))
    raw = set(db.setup_by_vs(vs=VS.raw))
    not_both_cpx_raw = set(cpx.symmetric_difference(raw))
    valid = set(db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.valid))
    confirmed = set(db.setup_by_data_validity(sensor=Sensor.nes, value=Validation.confirmed))
    recorded_here = set(db.setup_by_project(prj=Project.neteera))

    motion2 = set(db.setup_by_state(State.is_motion, value=True))
    motion = set(db.setup_vs_equals(VS.rest, 0))
    bp = set(db.setup_by_vs(VS.bp))
    speaking = set(db.setup_vs_equals(VS.speaking, 1))
    speaking2 = set(db.setup_by_state(State.is_speaking, value=True))
    sitting = set(db.setup_by_posture(Posture.sitting))
    gt_definition_amb = set(db.setup_by_note('Station A protocol'))
    inters = bp & fmcw & recorded_here & (back | front) & sitting & no_driving & no_driving2 & no_driving3 & (
             valid | confirmed) & not_both_cpx_raw - speaking - speaking2 - motion - motion2
    #valid | confirmed) & not_both_cpx_raw - ec - ec2 - speaking - speaking2 - motion - motion2 - zrr - zrr2

    # - set([3869, 4121, 3407, 3409, 3410, 3411, 3414, 3415, 4156, 3151, 3156, 3157, 3172, 3434, 3436, 4220,
             #        4408, 4381, 3664, 4340, 4178, 3456, 3449, 3165, 4285, 3800, 3282, 3519, 3528, 4160, 4341,
             #        3397, 3800, 4285, 4485, 4305, 3866, 3534, 4172, 4572, 4397, 4328, 4801])
    return inters


def zero_crossing(x: iter) -> int:
    """number of times the signal crosses the zero"""
    return ((x[:-1] * x[1:]) < 0).sum()

if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    hb_feature = []

    bp_labels = []
    names = []
    setup_ids = []

    setups_bp = setuplist()
    #print(set(db.setup_from_to_date(from_date='2021-05-31 10:25:00', to_date='2022-05-31 11:00:00')))
    setups_bp = setups_bp - set(db.setup_from_to_date(from_date='2021-08-24 06:25:00', to_date='2027-05-31 11:00:00'))

    setups = []
    new_fs = args.fs if args.fs is not None else 500
    for s in setups_bp:
        if db.setup_ref_path_npy(setup=s, sensor=Sensor.epm_10m, vs=VS.bp) != None:
            try:
                np.load(db.setup_ref_path_npy(setup=s, sensor=Sensor.epm_10m, vs=VS.bp), allow_pickle=True)
                # print(np.load(db.setup_ref_path_npy(setup=s, sensor=Sensor.epm_10m, vs=VS.bp), allow_pickle=True))
            except:
                continue
                print("setup", s, "not ok")
            setups.append(s)

    back = list(set(setups) & set(db.setup_by_target(target=Target.back)))
    front = list(set(setups) & set(db.setup_by_target(target=Target.front)))

    stress_setups = db.setup_by_note(note='Stress')

    s_vals = []

    for idx in front:
        #if db.setup_subject(idx) == 'linda':
        #    continue
        print(idx,db.setup_subject(idx), db.setup_distance(idx))
        X = []
        y = []
        zcs = []
        path = db.setup_ref_path(idx, Sensor.nes)
        # data = load_nes(path[0], NeteeraSensorType(NeteeraSensorType(db.setup_mode(idx))))
        data = load_nes(idx, db)
        fs = data['framerate']
        param_path = db.setup_ref_path(setup=idx, sensor=Sensor.epm_10m)
        param_file = pd.read_csv(param_path[0])

        bp_data = param_file.drop(param_file[param_file[' NIBP-S(mmHg)'] == '--'].index)

        if len(bp_data) == 0:
            print("no bp data")
            continue
        print(bp_data[' NIBP-S(mmHg)'])
        bp_idx = np.zeros((len(bp_data),  args.nsec * new_fs))
        bp_values = np.zeros((len(bp_data), 2))
        for i in bp_data.index:
            print(i)
            print(np.arange(i * new_fs, (i + args.nsec) * new_fs))
            # dt_bp = datetime.datetime.strptime(bp_data.Time[i], '%Y-%m-%d %H:%M:%S')
            # t = dt_bp - delay

            bp_idx[np.where(bp_data.index == i)[0][0]] = np.arange(i * new_fs, (i + args.nsec) * new_fs)
            bp_values[np.where(bp_data.index == i)[0][0]] = bp_data[' NIBP-S(mmHg)'][i], bp_data[' NIBP-D(mmHg)'][i]
        s_vals.append(bp_data[' NIBP-S(mmHg)'].to_numpy())


        sd = np.max(bp_values[:, 0]) - np.min(bp_values[:, 0])
        dd = np.max(bp_values[:, 1]) - np.min(bp_values[:, 1])


        if 'stress' in db.setup_note(idx):
            print("stress")
        sloc = np.argsort(bp_values[:,0])
        lowest = min(2, len(bp_data))

        distance = db.setup_distance(idx)
        C = 3e8
        bandwidth = data['BW'] * 1e6  # MHz
        fs = db.setup_fs(idx) * 1e6  # MS/s
        chirp_duration = 68 / fs
        df = fs / 256
        alpha = bandwidth / chirp_duration
        dr = 1000 * (df * C / 2 / alpha)
        trg_rng_bin = int(distance / dr)
        bin_by_dist = np.argmin(np.abs(trg_rng_bin - data['bins']))
        bin_by_argmax = np.argmax(np.var(np.abs(data['data']), axis=0))
        ok_bins = []
        # for b in range(data['data'].shape[1]):
        #     z = data['data'].to_numpy()[:, b]
        #
        #     zc = zero_crossing(z)/len(z)
        #     print(b, zc, np.var(z))
        #
        #     if zc > 0.99 or zc < 0.175:
        #         continue
        #     ok_bins.append([b, zc])
        # ok_bins = np.stack(ok_bins)
        # print(ok_bins)
        # bin = int(ok_bins[np.argmin(ok_bins[:,1]),0])
        #
        # print("-----------------")
        # z = data['data'].to_numpy()[:, bin]
        # zc = zero_crossing(z)/len(z)
        # print(bin, zc)
        # print(bin_by_dist)
        try:
            dat = data['data'].to_numpy()[:, bin_by_argmax]
        except:
            print("???")
        print(1)
        dat = resample_sig(dat.T, int(db.setup_fs(idx)), new_fs).T

        x, hb_idx = front_features(dat, new_fs, normalize=False)
        print(2)
        nsec = args.nsec
        feature_size = new_fs * nsec


        for bp_row in bp_data.iterrows():
            sec = bp_row[0]
            systolic =  bp_row[1][' NIBP-S(mmHg)' ]
            diastolic =  bp_row[1][' NIBP-D(mmHg)' ]
            start_idx = sec*new_fs
            end_idx = start_idx + nsec*new_fs
            y.append([bp_values[reading][0], bp_values[reading][1]])
            X.append(x[:, hb_start:hb_end])
            names.append(db.setup_subject(setup=idx))
            setup_ids.append(idx)

            if end_idx >= x.shape[1]:
                continue
        for j in hb_idx:  # iterate over individual heartbeats, skip the first and the last until the edges problem solved
            hb_start = j
            hb_end = j + nsec*new_fs
            if hb_end >= x.shape[1]:
                continue
            #print(hb_start, hb_end, hb_start in bp_idx, hb_end in bp_idx)

            if hb_start in bp_idx and hb_end in bp_idx:

                reading = np.where(bp_idx == j)[0][0]
                #print(bp_values[reading])
                orig_x = dat[hb_start:hb_end]
                orig_x_deterend = orig_x - np.median(orig_x)
                zc = zero_crossing(orig_x_deterend)/len(orig_x_deterend)
                zcs.append(zc)


                y.append([bp_values[reading][0], bp_values[reading][1]])
                X.append(x[:, hb_start:hb_end])

                #print(j, x[:, hb_start:hb_end].shape)
                names.append(db.setup_subject(setup=idx))
                setup_ids.append(idx)


        #np.save(os.path.join(args.save_path, str(idx)+ '_X.npy'), hb_feature)
        if X:
            X = np.stack(X)
            y = np.stack(y)
            zcs = np.stack(zcs)


            np.save(os.path.join(args.save_path, str(idx)+ '_X.npy'), X, allow_pickle=True)
            np.save(os.path.join(args.save_path, str(idx)+ '_y.npy'), y, allow_pickle=True)
            np.save(os.path.join(args.save_path, str(idx)+ '_zc.npy'), zcs, allow_pickle=True)
            print(idx, "done")
        #np.save(os.path.join(args.save_path, str(idx)+ '_y.npy'), bp_labels)
        #np.save(os.path.join(args.save_path, str(idx)+ '_name.npy'), names)
        #np.save(os.path.join(args.save_path, 'setups.npy'), setup_ids)
