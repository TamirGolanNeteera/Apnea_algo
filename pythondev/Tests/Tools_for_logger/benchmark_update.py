# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
from Tests.Utils.DBUtils import calculate_delay
from Tests.Utils.LoadingAPI import load_reference
from Tests.Utils.ResearchUtils import print_var
from Tests.Tools_for_logger.GenerateRrReferenceFromNatus import calc_rr_from_ref
from Tests.vsms_db_api import *


if __name__ == '__main__':

    db = DB('neteera_cloud_mirror')
    benchmark_functions = {
        'med_bridge': db.setup_med_bridge_benchmark,
        # 'fae_rest': db.setup_fae_rest,
        # 'ec_benchmark': db.setup_ec_benchmark,
        # 'mild_motion': db.setup_mild_motion_benchmark,
        # 'ie_bench': db.setup_ie_benchmark,
        # 'nwh': db.setup_nwh_benchmark,
        # 'cen_exel': db.setup_cen_exel_benchmark,
    }

    for bench, func in benchmark_functions.items():
        print_var(bench)
        new = func()
        old = db.benchmark_setups(bench)

        setups_to_remove = set(old) - set(new)
        setups_to_add = set(new) - set(old)

        for s in setups_to_remove:
            db.set_benchmark(setup=s, benchmark=Benchmark[bench], value=False)
        print_var(setups_to_remove)

        if bench == 'nwh':
            for s in setups_to_add:
                load_reference(s, ['apnea', 'posture', 'sleep_stages'], use_saved_npy=False)
                calculate_delay(s, 'hr', db, False)
                calc_rr_from_ref(s, db)

        for s in setups_to_add:
            print(db.set_benchmark(setup=s, benchmark=Benchmark[bench], value=True))
        print_var(setups_to_add)

        assert set(new) == set(db.benchmark_setups(bench))
