from datetime import datetime

import pandas as pd
import pytz
import numpy as np
import json
#from vsms_DB_API.db_constants import Sensor, VS
from pathlib import Path
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()
from Tests.Utils.LoadingAPI import load_reference
#from local_DB_API.vsms_db_api import DB
from Tests.vsms_db_api import *
import scipy.signal as sp

db = DB('neteera_cloud_mirror')
import mne
from datetime import timedelta
import matplotlib.pyplot as plt
from create_apnea_count_AHI_data import compute_respiration, apnea_class
from create_apnea_count_AHI_data_regression_from_storage import create_AHI_regression_training_data_MB_phase_with_sleep_ref



def get_gap_in_frames(db, setup, fs=500):
    setup_dir, sess_dir = list(Path(db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)).parents)[1:3]
    ref_json_file = list(sess_dir.rglob('*ref_metadata.json'))[0]
    radar_json_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
    edf_path = list(setup_dir.parent.rglob('*edf'))[0]
    with open(ref_json_file, 'r') as file:
        radar_start_time_str = json.load(file)['StartDateAndTime']
        radar_start_time = datetime.datetime.fromisoformat(radar_start_time_str)

    ref_start_time = mne.io.read_raw_edf(edf_path).info['meas_date'] + timedelta(hours=4)
    if ref_start_time.replace(tzinfo=None) > datetime.datetime(year=2023, month=11, day=5):
        ref_start_time +=  timedelta(hours=1)
    delta_sec = abs(ref_start_time - radar_start_time)
    ref_earlier = ref_start_time > radar_start_time
    delta_in_hz = delta_sec.total_seconds() * fs
    return int(delta_in_hz), ref_earlier



def get_gap_in_frames_old(db, setup, fs=500):
    setup_dir, sess_dir = list(Path(db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)).parents)[1:3]
    try:
        ref_json_file = list(sess_dir.rglob('*ref_metadata.json'))[0]
    except:
        print("***")
    radar_json_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
    with open(ref_json_file, 'r') as file:
        ref_start_time_str = json.load(file)['StartDateAndTime']
        ref_start_time = datetime.datetime.fromisoformat(ref_start_time_str)
    with open(radar_json_file, 'r') as file:
        radar_start_time_str = json.load(file)['start_time']
        radar_start_time = datetime.datetime.fromisoformat(radar_start_time_str).replace(tzinfo=pytz.UTC)
    delta_sec = radar_start_time - ref_start_time
    delta_in_hz = delta_sec.total_seconds() * fs
    return int(delta_in_hz)


# def get_gap_in_frames(db, setup, fs=500):
#     setup_dir, sess_dir = list(Path(db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)).parents)[1:3]
#     ref_json_file = list(sess_dir.rglob('*ref_metadata.json'))[0]
#     radar_json_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
#     with open(ref_json_file, 'r') as file:
#         ref_start_time_str = json.load(file)['StartDateAndTime']
#         ref_start_time = datetime.fromisoformat(ref_start_time_str)
#     with open(radar_json_file, 'r') as file:
#         radar_start_time_str = json.load(file)['start_time']
#         radar_start_time = datetime.fromisoformat(radar_start_time_str).replace(tzinfo=pytz.UTC)
#     delta_sec = radar_start_time - ref_start_time
#     delta_in_hz = delta_sec.total_seconds() * fs
#     return int(delta_in_hz)

def setup_start_time(setup, start_time):
    #datetime_object = datetime.strptime(start_time, '%y-%m-%dT%H:%M:%S')
    return datetime.datetime.fromisoformat(start_time).replace(tzinfo=pytz.UTC)


def setup_end_time(setup, end_time):
    return datetime.datetime.fromisoformat(end_time).replace(tzinfo=pytz.UTC)


def load_and_process_setup_data(setup, phase, ref_start_time):
    delta = get_gap_in_frames(setup)
    phase_data = np.load(phase)

    # Cut along the reference start time
    ref_delta = int(delta / 500)  # Convert frames to seconds
    phase_data = phase_data[ref_delta:]

    # Process the data as needed
    # ...


def stitch_and_align_setups_of_session(session=107978):
    device_map = {232:1, 238:1, 234:1, 236:1, 231:1,
                  240:2, 248:2, 254:2, 250:2, 251:2,
                  270:3, 268:3, 256:3, 269:3, 259:3,
                    278:4, 279:4, 273:4, 271:4, 274:4}
    db = DB('neteera_cloud_mirror')
    setups = db.setups_by_session(session)
    device_setups = {}
    setups_start_time = {}
    setups_end_time = {}
    setups_phase = {}
    statuses = {}
    final_data_dict = {}

    for setup in setups:
        setup_dir = Path(db.setup_dir(setup))
        raw_metadata_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
        stat_file = list(setup_dir.rglob('*stat.npy'))[0]
        with open(raw_metadata_file, 'r') as file:

            raw_metadata = json.load(file)
            device_id = raw_metadata.get('device_sn')
            setups_start_time[setup] = setup_start_time(setup, raw_metadata.get('start_time'))
            setups_end_time[setup] = setup_start_time(setup, raw_metadata.get('end_time'))

            # Store the phase file path for the setup
            # phase_file = db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)
            phase_file = db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)
            setups_phase[setup] = phase_file
            statuses[setup] = stat_file
            if device_id not in device_setups:
                device_setups[device_id] = []

            device_setups[device_id].append(setup)

    for device_id, setups_list in device_setups.items():
        setups_list.sort(key=lambda setup: setups_start_time[setup])
        merged_setups = []
        print(setups_list)
        merged_setup = setups_list[0]
        phase_merged = np.load(setups_phase[setups_list[0]])
        staus_merged = np.load(statuses[setups_list[0]], allow_pickle=True)
        for i in range(1, len(setups_list)):
            current_setup = setups_list[i]
            prev_setup = setups_list[i - 1]
            current_start_time = setups_start_time[current_setup]
            prev_end_time = setups_end_time[prev_setup]

            # Merge the phase data files
            phase_current = np.load(setups_phase[current_setup])
            stat_current = np.load(statuses[current_setup], allow_pickle=True)
            gap_frames = (current_start_time - prev_end_time).seconds * 500
            gap_sec = (current_start_time - prev_end_time).seconds
            if gap_frames > 0:
                gap_data = np.full(gap_frames, np.nan)
                gap_data_sec = np.full(gap_sec, np.nan)
                phase_merged = np.concatenate((phase_merged, gap_data, phase_current))
                staus_merged = np.concatenate((staus_merged, gap_data_sec, stat_current))
            else:
                phase_merged = np.concatenate((phase_merged, phase_current))
                staus_merged = np.concatenate((staus_merged, stat_current))

            # Load and process the merged data
        gap_from_ref = get_gap_in_frames(db, setups_list[0])
        device_location = device_map[int(device_id) % 1000]
        final_data_dict[tuple(setups_list)] = {'phase': phase_merged, 'gap': gap_from_ref, 'device_loc':device_location, 'stat':staus_merged}
    return final_data_dict

if __name__ == '__main__':

    sessions = [108168, 108139, 108145, 108146, 108147, 108148, 108152, 108153, 108154, 108159, 108168, 108170, 108171, 108175, 108186, 108191, 108192, 108201, 108202]
    for s in sessions:
        print("session", s,":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        data_dict = stitch_and_align_setups_of_session(session=s)#(session=107978)
        mb_dir = '/Neteera/Work/homes/dana.shavit/Research/apnea2021/prepared_data/stitched_2611/'
        if not os.path.isdir(mb_dir):
            os.makedirs(mb_dir)

        #print(data_dict)
        for k,v in data_dict.items():

            setup = min(k)
            print(db.setup_subject(setup), setup, ":::::::::::::::::::::")
            #print(k)
            ph = v['phase']

            gap = v['gap'][0]
            apnea_ref = load_reference(setup, 'apnea', db)
            ss_ref = load_reference(setup, 'sleep_stages', db)
            ph = pd.DataFrame(ph)
            ph.interpolate(method="linear", inplace=True)
            ph.fillna(method="bfill", inplace=True)
            ph.fillna(method="pad", inplace=True)

            ph = ph.to_numpy()
            respiration = compute_respiration(ph.flatten())
            gap = int(gap / 500)
            if v['gap'][1]:
                apnea_ref = apnea_ref[gap:]
                ss_ref = ss_ref[gap:]
            print("after crop len(apnea_ref)", len(apnea_ref), "len(phase)", len(ph) / 500, "len ss", len(ss_ref))
            fig, ax = plt.subplots(4, sharex=False, figsize=(14, 7))
            ax[0].plot(ph)
            ax[1].plot(apnea_ref)
            ax[2].plot(ss_ref)
            ax[3].plot(respiration)
            plt.show()



            UP = 1
            DOWN = 50
            respiration = sp.resample_poly(respiration, UP, DOWN)
            fs_new = int((500 * UP) / DOWN)
            apnea_ref_class = np.zeros(len(apnea_ref))

            for i in range(len(apnea_ref)):
                # print(i, apnea_ref[i], apnea_ref[i] in apnea_class.keys())
                if apnea_ref[i] not in apnea_class.keys():
                    apnea_ref_class[i] = -1
                else:
                    apnea_ref_class[i] = int(apnea_class[apnea_ref[i]])

            print(np.unique(apnea_ref_class))

            print(np.unique(ss_ref))
            ss_ref_class = np.zeros(len(ss_ref))
            ss_class = {'N1': 1, 'N2': 1, 'N3': 1, 'W': 0, 'R': 1}
            for i in range(len(ss_ref)):
                # print(i, ss_ref[i], ss_ref[i] in ss_class.keys())
                if ss_ref[i] not in ss_class.keys():
                    ss_ref_class[i] = -1
                else:
                    ss_ref_class[i] = int(ss_class[ss_ref[i]])

            print(np.unique(ss_ref_class))

            empty_ref_class = []
            chunk_size_in_minutes = 15
            time_chunk = fs_new * chunk_size_in_minutes * 60
            step = fs_new * 15 * 60

            try:
                X, y, valid, = create_AHI_regression_training_data_MB_phase_with_sleep_ref(respiration=respiration,
                                                                                           apnea_ref=apnea_ref_class,
                                                                                           sleep_ref=ss_ref_class,
                                                                                           empty_ref=empty_ref_class,
                                                                                           time_chunk=time_chunk,
                                                                                           step=step,
                                                                                           scale=args.scale,
                                                                                           fs=fs_new)
                print("np.unique(y)", np.unique(y), setup, db.session_from_setup(setup))
                print(y)
            except:
                print("oh crap")

            print(X.shape, y.shape)
            print("successfully created AHI labels")
            np.save(os.path.join(mb_dir, str(setup) + '_y.npy'), y, allow_pickle=True)
            np.save(os.path.join(mb_dir, str(setup) + '_X.npy'), X, allow_pickle=True)
            np.save(os.path.join(mb_dir, str(setup) + '_valid.npy'), valid, allow_pickle=True)
            np.save(os.path.join(mb_dir, str(setup) + '_empty_ref_class.npy'), empty_ref_class, allow_pickle=True)
            np.save(os.path.join(mb_dir, str(setup) + '_ss_ref_class.npy'), ss_ref_class, allow_pickle=True)
            np.save(os.path.join(mb_dir, str(setup) + '_apnea_ref_class.npy'), apnea_ref_class, allow_pickle=True)
            print("saved training data")

