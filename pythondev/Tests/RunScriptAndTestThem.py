from Tests.Utils.TestsUtils import run_cmd
import os

if __name__ == '__main__':
    pythondev_dir = os.path.dirname(os.path.dirname(__file__))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'Tester.py -start_time s60'))

    run_cmd(os.path.join(
        pythondev_dir, 'Tests', 'Plots',
        'PlotRawDataRadarCPX.py -distance 1000 -tlog_paths '
        '/Neteera/Work/homes/moshe.caspi/projects/210/gain_tests/1652877279/_165287727990.ttlog'
        ' -plots iq iq_vs_time amplitude displacement'))
    run_cmd(os.path.join(
        pythondev_dir, 'Tests', 'Plots',
        'PlotRawDataRadarCPX.py -distance 1000 -json_paths '
        '/Neteera/Work/homes/moshe.caspi/data/Herzog/rr_under_5/'
        'd9832658-047a-48e3-815e-4684977ce37e_1652234564_1652234710/b215841c-2bc9-4f18-9243-fd21a1ee3ffe/'
        'b215841c-2bc9-4f18-9243-fd21a1ee3ffe_1652964077_cloud_raw_data.json'))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'Plots', 'PlotRawDataRadarCPX.py -distance 1000 -setups 9710'))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'VersionTester.py -benchmark ec_benchmark'))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'Plots', 'plot_raw_data_from_cloud.py'))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'Plots', 'PlotEPM10m.py -setup_id 5721'))

    run_cmd(os.path.join(pythondev_dir, 'Tests', 'Plots',
                         'BitExactPythonCPP.py -py'
                         '/Neteera/Work/NeteeraVirtualServer/DELIVERED/Algo/net-alg-3.5.9/stats/ec_benchmark'
                         '-cpp /Neteera/Work/homes/nachum_shtauber/Neteera/results/VER_1_23_2_0_again_ec_benchmark'))



