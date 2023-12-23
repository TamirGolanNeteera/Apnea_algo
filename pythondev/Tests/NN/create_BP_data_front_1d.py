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


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=False)
    parser.add_argument('-fs', metavar='seed', type=int, required=False, help='New sampling rate')
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save data', required=True)
    parser.add_argument('-target', metavar='Target', type=str, help='back or front', required=True)
    parser.add_argument('--clean_train', action='store_true', help='clean high variance sessions from training set')
    return parser.parse_args()


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

    return inters


if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    hb_feature = []
    noisy_hb_feature = []
    bp_labels = []
    names = []
    setup_ids = []
    noisy_bp_labels = []
    noisy_names = []
    noisy_setup_ids = []
    setups_bp = setuplist()
    #print(set(db.setup_from_to_date(from_date='2021-05-31 10:25:00', to_date='2022-05-31 11:00:00')))
    setups_bp = setups_bp - set(db.setup_from_to_date(from_date='2021-08-24 06:25:00', to_date='2027-05-31 11:00:00'))

    setups = []
    new_fs = args.fs if args.fs is not None else 500
    for s in setups_bp:
        if db.setup_ref_path_npy(setup=s, sensor=Sensor.epm_10m, vs=VS.bp) != None:
            try:
                np.load(db.setup_ref_path_npy(setup=s, sensor=Sensor.epm_10m, vs=VS.bp), allow_pickle=True)
            except:
                continue
                print("setup", s, "not ok")
            setups.append(s)

    back = list(set(setups) & set(db.setup_by_target(target=Target.back)))
    front = list(set(setups) & set(db.setup_by_target(target=Target.front)))

    stress_setups = db.setup_by_note(note='Stress')

    print("front", front)
    print(len(front))
    ec2 = set(db.setup_by_state(state=State.is_empty, value=True))
    ec = set(db.setup_vs_equals(VS.occupancy, 0))
    zrr2 = set(db.setup_by_state(state=State.is_hb, value=True))
    zrr = set(db.setup_vs_equals(VS.zrr, 1))


    for idx in front:
        print(idx)
        X = []
        y = []
        path = db.setup_ref_path(idx, Sensor.nes)
        # data = load_nes(path[0], NeteeraSensorType(NeteeraSensorType(db.setup_mode(idx))))
        data = load_nes(idx, db)
        fs = data['framerate']
        param_path = db.setup_ref_path(setup=idx, sensor=Sensor.epm_10m)
        param_file = pd.read_csv(param_path[0])
        try:
            delay = calculate_delay(idx, 'hri', db)


        except (Exception, AssertionError, FileNotFoundError, TypeError, IndexError) as err:
            print(err)
            continue

        bp_data = param_file.drop(param_file[param_file[' NIBP-S(mmHg)'] == '--'].index)
        print(bp_data)
        if len(bp_data) == 0:
            continue

        bp_idx = np.zeros((len(bp_data), 30 * new_fs))
        bp_values = np.zeros((len(bp_data), 2))
        for i in bp_data.index:
            # dt_bp = datetime.datetime.strptime(bp_data.Time[i], '%Y-%m-%d %H:%M:%S')
            # t = dt_bp - delay
            t = i - delay  # make sure it is the right direction
            bp_idx[np.where(bp_data.index == i)[0][0]] = np.arange((t - 30) * new_fs, t * new_fs)
            bp_values[np.where(bp_data.index == i)[0][0]] = bp_data[' NIBP-S(mmHg)'][i], bp_data[' NIBP-D(mmHg)'][i]

        sd = np.max(bp_values[:, 0]) - np.min(bp_values[:, 0])
        dd = np.max(bp_values[:, 1]) - np.min(bp_values[:, 1])
        if args.clean_train and (idx in stress_setups):
            print("stress or high variance, discarding")
            continue
        mean_sys = np.mean(bp_values[:, 0])
        mean_dias = np.mean(bp_values[:, 1])

        sloc = np.argsort(bp_values[:,0])
        lowest = min(2, len(bp_data))

        #print(bp_idx)
        s_to_use = mean_sys#np.mean(bp_values[sloc[:lowest],0])
        d_to_use = mean_dias#np.mean(bp_values[sloc[:lowest],1])


        distance = db.setup_distance(idx)
        C = 3e8
        bandwidth = data['BW'] * 1e6  # MHz
        fs = db.setup_fs(idx) * 1e6  # MS/s
        chirp_duration = 68 / fs
        df = fs / 256
        alpha = bandwidth / chirp_duration
        dr = 1000 * (df * C / 2 / alpha)
        trg_rng_bin = int(distance / dr)
        bin = np.argmin(np.abs(trg_rng_bin - data['bins']))
        dat = data['data'].to_numpy()[:, bin]
        if  int(db.setup_fs(idx)) != new_fs:
            dat = resample_sig(dat.T, int(db.setup_fs(idx)), new_fs).T
        x, hb_idx = front_features(dat, new_fs, normalize=False)

        nsec = 10
        feature_size = new_fs * nsec


        for j in hb_idx:  # iterate over individual heartbeats, skip the first and the last until the edges problem solved
            hb_start = j
            hb_end = j + nsec*new_fs
            if hb_end >= x.shape[1]:
                continue
            #print(hb_start, hb_end, hb_start in bp_idx, hb_end in bp_idx)

            if hb_start in bp_idx and hb_end in bp_idx:

                reading = np.where(bp_idx == j)[0][0]
                y.append([bp_values[reading][0], bp_values[reading][1]])
                X.append(x[:, hb_start:hb_end])

                #print(j, x[:, hb_start:hb_end].shape)
                names.append(db.setup_subject(setup=idx))
                setup_ids.append(idx)


        #np.save(os.path.join(args.save_path, str(idx)+ '_X.npy'), hb_feature)
        if X:
            X = np.stack(X)
            y = np.stack(y)
            print(idx, X.shape, y.shape)
            np.save(os.path.join(args.save_path, str(idx)+ '_X.npy'), X, allow_pickle=True)
            np.save(os.path.join(args.save_path, str(idx)+ '_y.npy'), y, allow_pickle=True)
        #np.save(os.path.join(args.save_path, str(idx)+ '_y.npy'), bp_labels)
        #np.save(os.path.join(args.save_path, str(idx)+ '_name.npy'), names)
        #np.save(os.path.join(args.save_path, 'setups.npy'), setup_ids)