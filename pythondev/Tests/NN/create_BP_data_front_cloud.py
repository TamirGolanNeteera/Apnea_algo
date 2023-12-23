import argparse
import datetime
import pandas as pd
from sklearn.preprocessing import scale
import random
import glob
from Tests.NN.reiniers_create_data import *
from Tests.vsms_db_api import *
from Tests.Utils.DBUtils import calculate_delay
from Tests.Utils.LoadingAPI import load_nes
from pylibneteera.sp_utils import resample_sig
from local_DB_API.vsms_db_api import DB

def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-nsec', metavar='seed', type=int, required=False, help='Signal length in seconds')
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save data', required=True)
    parser.add_argument('-setups', '-session_ids', metavar='ids', nargs='+', required=False, type=int,
                        help='Index of setup in list')
    return parser.parse_args()




if __name__ == '__main__':
    db = DB('neteera_cloud_mirror')
    args = get_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    hb_feature = []
    bp_labels = []

    setups = args.setups

    for s in setups:
        files = db.setup_ref_path_npy(s, sensor=Sensor.epm_10m)
        if files != None:
            try:
                art_files = [f for f in files if 'Art' in f]
                if art_files != None:
                    art_s = np.load([f for f in files if 'Art-S' in f][0], allow_pickle=True)
                    art_d = np.load([f for f in files if 'Art-D' in f][0], allow_pickle=True)
                    art_m = np.load([f for f in files if 'Art-M' in f][0], allow_pickle=True)
                cuff_files = [f for f in files if 'NIBP' in f]
                if cuff_files != None:
                    nibp_s = np.load([f for f in files if 'NIBP-S' in f][0], allow_pickle=True)
                    nibp_d = np.load([f for f in files if 'NIBP-D' in f][0], allow_pickle=True)
                    nibp_m = np.load([f for f in files if 'NIBP-M' in f][0], allow_pickle=True)
                print("setup", s, "bp loaded")
                art_s_ok = []
                art_d_ok = []
                art_m_ok = []
                ref_time_index = np.load([f for f in files if 'time_index' in f][0], allow_pickle=True)

                radar_path = db.setup_dir(s)
                if 'NES_RAW' in os.listdir(radar_path):
                    nes_path = os.path.join(radar_path, 'NES_RAW')
                    nes_files = os.listdir(nes_path)
                    if nes_files:
                        ts_file = [f for f in nes_files if 'TIME_INDEX' in f][0]
                        time_index = np.load(os.path.join(nes_path, ts_file), allow_pickle=True)
                        cpx_file = [f for f in nes_files if '_cpx' in f][0]
                        cpx = np.load(os.path.join(nes_path, cpx_file), allow_pickle=True)
                        phase_file = [f for f in nes_files if '_phase' in f][0]
                        phase = np.load(os.path.join(nes_path, phase_file), allow_pickle=True)

                radar_files = glob.glob(radar_path)
                print(radar_files)
                time_index_dt64 = [td.astype('datetime64[D]') for td in time_index]
                ref_time_index_dt64 = [td.astype('datetime64[D]') for td in ref_time_index]
                for i, a in enumerate(art_s):
                    if type(a) == str:
                        art_ok.append([ref_time_index[i], int(art_s[i]), int(art_d[i]), int(art_m[i])])

            except:
                continue
                print("setup", s, "bp not ok")
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
    s_vals = []

    for idx in front:
        print(idx,db.setup_subject(idx))
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
        s_vals.append(bp_data[' NIBP-S(mmHg)'].to_numpy())



        sd = np.max(bp_values[:, 0]) - np.min(bp_values[:, 0])
        dd = np.max(bp_values[:, 1]) - np.min(bp_values[:, 1])

        mean_sys = np.mean(bp_values[:, 0])
        mean_dias = np.mean(bp_values[:, 1])
        dist_from_mean = []
        sys_vals = []
        for v in bp_values[:, 0]:
            sys_vals.append(v)
            dist_from_mean.append(np.round(np.abs(mean_sys-v), 2))
        print(sys_vals)
        print(dist_from_mean)
        if 'stress' in db.setup_note(idx):
            print("stress")
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
        if True:#new_fs != db.setup_fs(idx):
            #print("resampling..")
            #print(dat.shape)
            dat = resample_sig(dat.T, int(db.setup_fs(idx)), new_fs).T
            #print(dat.shape)
        x, hb_idx = front_features(dat, new_fs, normalize=False)

        nsec = args.nsec
        feature_size = new_fs * nsec

        for j in hb_idx:  # iterate over individual heartbeats, skip the first and the last until the edges problem solved
            hb_start = j
            hb_end = j + nsec*new_fs
            if hb_end >= x.shape[1]:
                continue
            #print(hb_start, hb_end, hb_start in bp_idx, hb_end in bp_idx)

            if hb_start in bp_idx and hb_end in bp_idx:

                reading = np.where(bp_idx == j)[0][0]
                #print(bp_values[reading])
                if 'stress' not in db.setup_note(idx) and np.abs(bp_values[reading][0] -mean_sys) > 15:
                    continue
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
