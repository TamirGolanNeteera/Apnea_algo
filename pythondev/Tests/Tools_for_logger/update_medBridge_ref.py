from Tests.Tools_for_logger.MedBridge_edf import edf_to_npy
from Tests.Utils.DBUtils import shift_reference
from Tests.Utils.LoadingAPI import load_reference
from Tests.Utils.IOUtils import load
from Tests.vsms_db_api import *


if __name__ == '__main__':
    db = DB('neteera_cloud_mirror')
    setups = db.benchmark_setups(Benchmark.med_bridge)

    for setup in setups:
        setup = 9838
        print(setup)
        edf_path = db.setup_ref_path(setup, Sensor.respironics_alice6, search='edf')[0]
        ret_edf = edf_to_npy(edf_path, save=True)
        for channel, new_ref_path in ret_edf['filenames'].items():
            if new_ref_path.endswith('.npy'):
                new_ref = load(new_ref_path)
                new_ref_shifted = shift_reference(setup, 'hr', new_ref, db)