from datetime import datetime
import pytz
import numpy as np
import json
from local_DB_API.db_constants import Sensor, VS
from pathlib import Path
from datetime import timedelta
import mne
from local_DB_API.vsms_db_api import DB
import pandas as pd


def get_gap_in_frames(db, setup, fs=500):
    setup_dir, sess_dir = list(Path(db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)).parents)[1:3]
    ref_json_file = list(sess_dir.rglob('*ref_metadata.json'))[0]
    # radar_json_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
    edf_path = list(setup_dir.parent.rglob('*edf'))[0]
    with open(ref_json_file, 'r') as file:
        radar_start_time_str = json.load(file)['StartDateAndTime']
        radar_start_time = datetime.fromisoformat(radar_start_time_str)

    ref_start_time = mne.io.read_raw_edf(edf_path).info['meas_date'] + timedelta(hours=4)
    if ref_start_time.replace(tzinfo=None) > datetime(year=2023, month=11, day=5):
        ref_start_time += timedelta(hours=1)
    delta_sec = abs(ref_start_time - radar_start_time)
    ref_earlier = ref_start_time < radar_start_time
    delta_in_hz = delta_sec.total_seconds() * fs
    print(ref_earlier)
    return int(delta_in_hz), ref_earlier


def setup_start_time(setup, start_time):
    return datetime.fromisoformat(start_time).replace(tzinfo=pytz.UTC)


def setup_end_time(setup, end_time):
    return datetime.fromisoformat(end_time).replace(tzinfo=pytz.UTC)


def load_and_process_setup_data(setup, phase, ref_start_time):
    delta = get_gap_in_frames(setup)
    phase_data = np.load(phase)

    # Cut along the reference start time
    ref_delta = int(delta / 500)  # Convert frames to seconds
    phase_data = phase_data[ref_delta:]

    # Process the data as needed
    # ...


def stitch_and_align_setups_of_session(session=107978, phase_dir=None):
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
    setups_hr = {}
    final_data_dict = {}
    statuses = {}


    for setup in setups:
        setup_dir = Path(db.setup_dir(setup))
        raw_metadata_file = list(setup_dir.rglob('*raw_metadata.json'))[0]
        stat_file = list(setup_dir.rglob('*stat.npy'))[0]
        with open(raw_metadata_file, 'r') as file:
            raw_metadata = json.load(file)
            device_id = raw_metadata.get('device_sn')
            setups_start_time[setup] = setup_start_time(setup, raw_metadata.get('start_time')) - timedelta(seconds=9)
            setups_end_time[setup] = setup_start_time(setup, raw_metadata.get('end_time'))

            # Store the phase file path for the setup
            # phase_file = db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)
            if phase_dir:
                phase_file = list(Path(phase_dir).rglob(f'*{setup}_RR_phase*'))[0]
            else:
                phase_file = db.setup_ref_path_npy(setup, Sensor.nes, vs=VS.phase)
            hr_file = list(Path(db.setup_dir(setup)).rglob(f'*hr*'))[0]
            setups_phase[setup] = phase_file
            statuses[setup] = stat_file
            setups_hr[setup] = hr_file

            if device_id not in device_setups:
                device_setups[device_id] = []

            device_setups[device_id].append(setup)

    for device_id, setups_list in device_setups.items():
        setups_list.sort(key=lambda setup: setups_start_time[setup])
        merged_setups = []

        merged_setup = setups_list[0]
        if phase_dir:
             phase_merged = np.stack(pd.read_pickle(setups_phase[setups_list[0]]).to_numpy()).flatten()
        else:
            phase_merged = np.load(setups_phase[setups_list[0]], allow_pickle=True)
        staus_merged = np.load(statuses[setups_list[0]], allow_pickle=True)
        hr_merged = np.load(setups_hr[setups_list[0]], allow_pickle=True)
        gaps_between_setups_arr = []
        for i in range(1, len(setups_list)):
            current_setup = setups_list[i]
            prev_setup = setups_list[i - 1]
            current_start_time = setups_start_time[current_setup]
            prev_end_time = setups_end_time[prev_setup]

            # Merge the phase data files
            if phase_dir:
                phase_current = np.stack(pd.read_pickle(setups_phase[current_setup]).to_numpy()).flatten()
            else:
                phase_current = np.load(setups_phase[current_setup])
            stat_current = np.load(statuses[current_setup], allow_pickle=True)
            hr_current = np.load(setups_hr[current_setup], allow_pickle=True)
            gap_frames = (current_start_time - prev_end_time).seconds * 500
            gap_sec = (current_start_time - prev_end_time).seconds
            if gap_frames > 0:
                gap_data = np.full(gap_frames, np.nan)
                gap_data_sec = np.full(gap_sec, np.nan)
                phase_merged = np.concatenate((phase_merged, gap_data, phase_current))
                staus_merged = np.concatenate((staus_merged, gap_data_sec, stat_current))
                hr_merged = np.concatenate((hr_merged, gap_data_sec, hr_current))
            else:
                phase_merged = np.concatenate((phase_merged, phase_current))
                staus_merged = np.concatenate((staus_merged, stat_current))
                hr_merged = np.concatenate((hr_merged, hr_current))

            # Load and process the merged data
        gap_from_ref, ref_earlier = get_gap_in_frames(db, setups_list[0])
        device_location = device_map[int(device_id) % 1000]
        final_data_dict[tuple(setups_list)] = {'phase': phase_merged, 'gap': gap_from_ref,
                                               'device_loc': device_location, 'ref_earlier' : ref_earlier, 'stat':staus_merged,
                                               'hr': hr_merged}
    return final_data_dict
