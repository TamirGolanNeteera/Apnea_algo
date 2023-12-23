from Tests.Tools_for_logger.NHW_edf import edf_to_npy
from Tests.Tools_for_logger.MedBridge_edf import edf_to_npy as read_edf_med_bridge
from Tests.Utils.DBUtils import shift_reference
from Tests.Utils.LoadingAPI import load_reference
from Tests.Utils.IOUtils import load
from Tests.vsms_db_api import *


if __name__ == '__main__':
    db = DB()
    setups = db.benchmark_setups(Benchmark.nwh)

    for setup in setups:
        print(setup)
        edf_path = db.setup_ref_path(setup, Sensor.natus, search='edf')[0]
        ret_edf = edf_to_npy(edf_path, save=True)
        for channel, new_ref_path in ret_edf['filenames'].items():
            if new_ref_path.endswith('.npy'):
                new_ref = load(new_ref_path)
                new_ref_shifted = shift_reference(setup, 'hr', new_ref, db)
                db.insert_npy_ref(setup, Sensor.natus, channel.lower(), new_ref_path, new_ref_shifted)
        for vs in ['posture', 'sleep_stages', 'apnea']:
            old = load_reference(setup, vs, db)
            reference_array = load_reference(setup, vs, db, use_saved_npy=False)
            new_ref_shifted = shift_reference(setup, 'hr', reference_array, db)
            npy_path = db.setup_ref_path_npy(setup, Sensor.natus, VS[vs])
            db.insert_npy_ref(setup, Sensor.natus, VS[vs], npy_path, new_ref_shifted)
