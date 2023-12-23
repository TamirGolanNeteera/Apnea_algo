import configparser

from Tests.vsms_db_api import *
from Tests.Utils.IOUtils import load


def load_fs(setup, db):
    fs_from_db = db.setup_fs(setup)
    if fs_from_db is None:
        metadata = load(db.setup_ref_path(setup, Sensor.nes)[0])['session_metadata']
        return metadata['fs']
    else:
        return fs_from_db


def load_baseFreq(setup, db):
    radar_config_from_db = db.setup_radar_config(setup)
    if radar_config_from_db:
        return radar_config_from_db['frontendConfig_baseFreq']
    else:
        config_path = db.setup_ref_path(setup, Sensor.nes, search='radarConfig')[0]
        config = configparser.RawConfigParser()
        config.read(config_path)[0]
        return config['RFE_CONFIG'].get('BaseFreq').split(' ')[0]


def load_PLLConfig_bandwidth(setup, db):
    radar_config_from_db = db.setup_radar_config(setup)
    if radar_config_from_db:
        return radar_config_from_db['PLLConfig_bandwidth']
    else:
        config_path = db.setup_ref_path(setup, Sensor.nes, search='radarConfig')[0]
        config = configparser.RawConfigParser()
        config.read(config_path)
        return config['PLL_CONFIG'].get('bandwidth').split(' ')[0]


if __name__ == '__main__':
    print(load_baseFreq(9803, DB()))