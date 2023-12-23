# Copyright (c) 2019 Neteera Technologies Ltd. - Confidential
# This script is used for the dblogger to show the reference data to the FAE

from Tests.Utils.DBUtils import *
from Tests.Plots.PlotRawDataRadarCPX import *
from Tests.Plots.PlotResults import plotter, performance
from Tests.Tools_for_logger.reference_data import load_data
from Tests.Utils.TestsUtils import shift_reliability, match_lists


import argparse
from argparse import RawTextHelpFormatter
from matplotlib import pyplot as plt


ymin_dict = {'hr': 40, 'spot_hr': 40, 'rr': 1, 'inhale_exhale': 0, 'inhale_time': 0, 'exhale_time': 0, 'hrv': 1,
             'ra': 1,
             'occupancy': -1, 'zrr': -1, 'speaking': -1, 'motion': -1, 'hri': 1}
ymax_dict = {'hr': 200, 'spot_hr': 200, 'rr': 40, 'inhale_exhale': 3, 'inhale_time': 3, 'exhale_time': 3, 'hrv': 500,
             'ra': 150,
             'occupancy': 3,
             'zrr': 3, 'speaking': 3, 'motion': 3, 'hri': 2000, 'hrv_nni_mean': 2000, 'hrv_sdnn': 500,
             'hrv_hf_max_power': 0.1,
             'hrv_hf_auc': 0.02}


def args() -> argparse.Namespace:
    """ Parse arguments"""
    parser = argparse.ArgumentParser(description='Plot reference results', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-sensor_type', metavar='sensor_type', type=str, required=True)
    parser.add_argument('-path_to_file', metavar='ref_path', type=str, required=True)
    parser.add_argument('-dist', type=float, required=False, help='distance to subject for fm_cw', default=None)
    return parser.parse_args()


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


sensor_type_vital_sign_dict = {s: [] for s in sensor_types}

for s in sensor_types:
    for k in reference_order.keys():
        if s in reference_order[k]:
            sensor_type_vital_sign_dict[s].append(k)

# sensor_type_vital_sign_dict['NES'] = ['displacment_offset_not_tracked', 'iq_offset_not_tracked',
#                                             'iq_vs_time_offset_not_tracked', 'displacment_offset_tracked',
#                                             'iq_offset_tracked', 'iq_vs_time_offset_tracked']
sensor_type_vital_sign_dict['NES'] = ['displacment_offset_not_tracked', 'iq_offset_not_tracked',
                                      'iq_vs_time_offset_not_tracked']

ymin_dict = {'hr': 40, 'rr': 1, 'ie': 0, 'inhale_time': 0, 'exhale_time': 0, 'hrv': 1, 'ra': 1,
             'occupancy': -1, 'zrr': -1, 'speaking': -1, 'motion': -1, 'hri': 1, 'bp': 0}

ymax_dict = {'hr': 200, 'rr': 40, 'ie': 3, 'inhale_time': 3, 'exhale_time': 3, 'hrv': 500, 'ra': 500,
             'occupancy': 3, 'zrr': 3, 'speaking': 3, 'motion': 3, 'hri': 2000, 'hrv_nni_mean': 2000,
             'hrv_sdnn': 500, 'hrv_hf_max_power': 0.1, 'hrv_hf_auc': 0.02, 'bp': 200}

ylabel = {'displacment_offset_not_tracked': r'Displacement [\mu m]',
          'displacment_offset_tracked': r'Displacement [\mu m]', 'iq_offset_not_tracked': 'Quadrature',
          'iq_offset_tracked': 'Quadrature', 'iq_vs_time_offset_not_tracked': 'Counts',
          'iq_vs_time_offset_tracked': 'Counts', 'ra': 'Displacement [um]', 'rr': 'RR [bpm]', 'hr': 'HR [bpm]',
          'hri': 'HRI [ms]', 'hrv_nni_mean': 'HRV_NNI_MEAN [ms]', 'hrv_sdnn': 'HRV_SDNN [ms]',
          'hrv_hf_max_power': 'HRV_HF_MAX_POWER [ms^2]', 'hrv_hf_auc': 'HRV_HF_AUC [ms^2]',
          'hrv_nni_cv': 'HRV_NNI_CV [ms]', 'hrv_lf_max_power': 'HRV_LF_MAX_POWER [ms^2]',
          'hrv_lf_auc': 'HRV_LF_AUC [ms^2]', 'ie': 'inhale / exhale [au]', 'bp': '[mmHg]'}

xlabel = {'displacment_offset_not_tracked': 'Time [s]', 'displacment_offset_tracked': 'Time [s]',
          'iq_offset_not_tracked': r'In Phase', 'iq_offset_tracked': r'In\Phase',
          'iq_vs_time_offset_not_tracked': 'Time [s]', 'iq_vs_time_offset_tracked': 'Time [s]', 'ra': 'Time [s]',
          'rr': 'Time [s]', 'hr': 'Time [s]', 'hri': 'Time [ms]', 'hrv_nni_mean': 'Time [s]', 'hrv_sdnn': 'Time [s]',
          'hrv_hf_max_power': 'Time [s]', 'hrv_hf_auc': 'Time [s]', 'hrv_nni_cv': 'Time [s]',
          'hrv_lf_max_power': 'Time [s]', 'hrv_lf_auc': 'Time [s]', 'ie': 'Time [s]', 'bp': 'Time [s]'}

raw_column = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3),
              10: (4, 3), 11: (4, 3), 12: (4, 3), 13: (4, 4), 14: (4, 4), 15: (4, 4), 16: (4, 4), 17: (5, 4),
              18: (5, 4), 19: (5, 4), 20: (5, 4)}


def process_nes_data(nes_data, channel_select_params, gap):
    win_size = 10
    data = roll_data(nes_data, channel_select_params, gap, win_size, dont_track)
    i = np.real(data)
    q = np.imag(data)
    phases = np.unwrap(np.angle(data), axis=0)
    time, displacement = gen_disp_time(phases, gap)
    return i, q, time, displacement


def plot(sensor: str, data: Dict[str, np.ndarray]):
    vss = sensor_type_vital_sign_dict[sensor]
    subplots = len(vss)
    raw, column = raw_column[len(vss)]
    counter = 0
    for i in range(raw):
        for j in range(column):
            if counter >= subplots or counter >= len(vss):
                break
            plt.subplot(raw, column, counter + 1)
            plt.title('reference system:{}, vital sign {}'.format(sensor, vss[counter]))
            plt.grid(True)
            if 'NES' in sensor:
                if vss[counter] not in data.keys():
                    out = process_nes_data(data['data'][1][0], data['data'][1][1], data['data'][1][2])
                    data.update({vss[counter]: np.asarray([[out[2][ii], out[3][ii]] for ii in range(len(out[2]))])})
                    data.update({vss[counter + 1]: np.asarray([[out[0][ii], out[1][ii]] for ii in range(len(out[0]))])})
                    data.update({vss[counter + 2]: np.asarray(
                        [[out[2][ii], out[0][ii], out[1][ii]] for ii in range(len(out[2]))])})
            else:
                try:
                    plt.ylim(bottom=ymin_dict[vss[counter]], top=ymax_dict[vss[counter]])
                except KeyError:
                    continue
                try:
                    plt.xlim(left=0, right=len(data[vss[counter]][1]))
                except KeyError:
                    counter = counter + 1
                    continue
                except IndexError:
                    print('crashed, reference system:{}, vital sign {}'.format(sensor, vss[counter]))
            plt.xlabel(xlabel[vss[counter]])
            plt.ylabel(ylabel[vss[counter]])
            plt.ion()
            plt.show()
            if vss[counter] == 'ra' and sensor == 'BIOPAC':
                plt.plot(data[vss[counter]][1]*1000)
            else:
                if 'NES' in sensor:
                    if vss[counter].find('iq_vs_time') >= 0:
                        plt.plot([data[vss[counter]][i][0] for i in range(len(data[vss[counter]]))],
                                 [data[vss[counter]][i][1] for i in range(len(data[vss[counter]]))], 'o',
                                 label='I', markersize=1)
                        plt.plot([data[vss[counter]][i][0] for i in range(len(data[vss[counter]]))],
                                 [data[vss[counter]][i][2] for i in range(len(data[vss[counter]]))], 'o',
                                 label='Q', markersize=1)
                        plt.legend()
                    else:
                        if vss[counter].find('displacment') >= 0:
                            plt.plot([data[vss[counter]][i][0] for i in range(len(data[vss[counter]]))],
                                     [data[vss[counter]][i][1] for i in range(len(data[vss[counter]]))],
                                     markersize=1)
                        else:
                            dont_track = vss[counter][vss[counter].find('_offset') + 1:].find('not_tracked') >= 0
                            channel_select_params = data['data'][1][1]
                            if dont_track:
                                if channel_select_params is not None:  # FMCW
                                    plt.xlim(0, 65536)
                                    plt.ylim(0, 65536)
                                else:  # CW
                                    plt.xlim(0, 4096)
                                    plt.ylim(0, 4096)
                            else:
                                if channel_select_params is not None:  # FMCW
                                    plt.xlim(-32768, 32768)
                                    plt.ylim(-32768, 32768)
                                else:  # CW
                                    plt.xlim(-2048, 2048)
                                    plt.ylim(-2048, 2048)
                            plt.plot([data[vss[counter]][i][0] for i in range(len(data[vss[counter]]))],
                                     [data[vss[counter]][i][1] for i in range(len(data[vss[counter]]))], 'o',
                                     markersize=1)
                else:
                    if data[vss[counter]][1].dtype in ['complex128']:
                        plt.plot(np.real(data[vss[counter]][1]), np.imag(data[vss[counter]][1]))
                    else:
                        if vss[counter] == 'hri':
                            plt.plot(np.ediff1d(data[vss[counter]][1]))
                        else:
                            plt.plot(data[vss[counter]][1])
            counter = counter + 1
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.draw()
    plt.pause(0.001)
    # plt.savefig('C:\\db_support\\pics\\' + sensor)
    # plt.clf()


def plot_pred_ref(pred_data, ref_data, data_type):
    reliability = None
    if data_type in ['hr', 'rr']:
        pred = pred_data[data_type][1][data_type]
        reliability = pred_data[data_type][1][data_type+'_cvg']
    else:
        pred = pred_data[data_type][1][data_type]
    pred, ref, shift = match_lists(pred, ref_data[data_type][1])
    if reliability is not None:
        reliability = shift_reliability(reliability, len(ref), shift)
    plotter(prediction=pred,
            gtruth=ref,
            reliability=reliability,
            vital_sgn=data_type,
            ymin=ymin_dict[data_type],
            ymax=ymax_dict[data_type])


def plot_preds_refs(pred_data, ref_data):
    vss = list(pred_data.keys())
    subplots = len(vss)
    raw, column = raw_column[len(vss)]
    counter = 0
    for i in range(raw):
        for j in range(column):
            if counter >= subplots or counter >= len(vss):
                break
            plt.subplot(raw, column, counter + 1)
            data_type = vss[counter]
            counter = counter + 1
            reliability = None
            if data_type in ['hr', 'rr']:
                prediction = pred_data[data_type][1][data_type]
                reliability = pred_data[data_type][1][data_type + '_cvg']
            else:
                prediction = pred_data[data_type][1][data_type]
            prediction, ref, shift = match_lists(prediction, ref_data[data_type][1])
            if reliability is not None:
                reliability = shift_reliability(reliability, len(ref), shift)

            plt.xlim(left=0, right=len(prediction))
            plt.xlabel('Time [s]')
            ax = plt.gca()
            n_ticks = 10
            jumps = max(1, int(len(prediction) / n_ticks))  # when len(prediction) < 10 set tick every second
            ax.set_xticks(np.arange(0, len(prediction), jumps))
            ax.set_xticklabels(np.arange(0, len(prediction), jumps))

            if ref is not None:
                plt.plot(ref, label='RS', color='orange')
            if prediction is not None:
                plt.plot(prediction, label='NETEERA')
            if reliability.shape:  # not None
                plt.plot(np.array([r if r > 0 else np.nan for r in reliability]) * prediction, label='RELIABILITY',
                         color='green', linewidth=2)
            plt.grid(True)
            if np.any(ref > 0) and np.any(prediction > 0):
                ymin = min(min(prediction[prediction > 0]), min(ref[ref > 0]), 45)
                ymax = max(max(prediction[prediction < 210]), max(ref[ref < 210]), 140)
                plt.ylim(bottom=ymin, top=ymax)
            ylabel = {'hr': 'HR [bpm]', 'spot_hr': 'Spot HR [bpm]', 'rr': 'RR [bpm]',
                      'inhale_exhale': 'inhale / exhale [au]',
                      'inhale_time': 'inhale / RR [au]', 'exhale_time': 'exhale / RR [au]', 'hrv': 'HRV [ms]',
                      'hri': 'HRI [ms]', 'occupancy': 'Occupancy', 'zrr': 'Breath holding', 'speaking': 'Speaking',
                      'motion': 'Motion'}

            pref_text_dict = {'hr': True, 'spot_hr': True, 'rr': True, 'inhale_exhale': False, 'inhale_time': False,
                              'exhale_time': False,
                              'hrv': True, 'ra': True, 'hri': True, 'occupancy': True, 'zrr': True, 'speaking': True,
                              'motion': True}

            plt.ylabel(ylabel[data_type])
            if pref_text_dict[data_type]:
                performance_str = {'hr': '{:10.2f}% of the data in 10% range error',
                                   'rr': '{:10.2f}% of the data in 4bpm range error',
                                   'hri': '{:10.2f}% of the data in 15ms range error',
                                   'occupancy': '     False alarm: {:4.2f}%, Miss: {:4.2f}%',
                                   'zrr': '     False alarm: {:4.2f}%, Miss: {:4.2f}%',
                                   'speaking': '     False alarm: {:4.2f}%, Miss: {:4.2f}%',
                                   'motion': '     False alarm: {:4.2f}%, Miss: {:4.2f}%'}
                try:
                    txt = performance_str[data_type].format(performance(prediction, ref, reliability, data_type))
                except (TypeError, IndexError):
                    pref = performance(prediction, ref, reliability, data_type)
                    txt = performance_str[data_type].format(pref[0], pref[1], 10)
                plt.text(2, ymin, txt, fontsize=8)
            plt.legend()

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.draw()
    plt.pause(0.001)
    # plt.savefig('C:\\db_support\\pics\\' + sensor)
    # plt.clf()


if __name__ == '__main__':
    # try:
    #     datas = {}
    #     args = args()
    #     args.data_type = None
    #     datas = load_data(args, log=[])
    #     plot(args.sensor_type, datas)
    # except KeyError:
    #     print('sensor {} not supported'.format(args.sensor_type))


    ### PRED VS REF ###

    args = argparse.Namespace(
        path_to_file='/Neteera/DATA/2021/3/24/4812/6271/NES_RES/265/6271_265_NES_mqtt_results_0.1.4.4_1616592671952_10.csv',
        sensor_type="NES", data_type='rr')
    pred_data = load_data(args, log=[])

    # args = argparse.Namespace(
    #     path_to_file=r"S:\2020\9\6\3518\3790\REFERENCE\BIOPAC\3790_BIOPAC_Untitled1_(smooth_&_RR_wave).txt",
    #     sensor_type="BIOPAC", data_type='rr')
    # ref_data = load_data(args, log=[])
    #
    # plot_pred_ref(pred_data, ref_data, 'rr')

    ### PREDS VS REFS ###

    #     pred_data = {}
    #     ref_data = {}
    #
    #     args = argparse.Namespace(
    #         path_to_file=r"N:\moshe.aboud\example\results_1.10.4.9_1600875423968_10.csv",
    #         sensor_type="NES", data_type='hr')
    #     pred_data_hr = load_data(args, log=[])
    #     args = argparse.Namespace(
    #         path_to_file=r"N:\moshe.aboud\example\ParameterData-20200922151524.csv",
    #         sensor_type="EPM_10M", data_type='hr')
    #     ref_data_hr = load_data(args, log=[])
    #
    #     args = argparse.Namespace(
    #         path_to_file=r"N:\moshe.aboud\example\results_1.10.4.9_1600875423968_10.csv",
    #         sensor_type="NES", data_type='rr')
    #     pred_data_rr = load_data(args, log=[])
    #     args = argparse.Namespace(
    #         path_to_file=r"N:\moshe.aboud\example\ParameterData-20200922151524.csv",
    #         sensor_type="EPM_10M", data_type='rr')
    #     ref_data_rr = load_data(args, log=[])
    #
    # #    plot_pred_ref(pred_data_rr, ref_data_rr, 'rr')
    #
    #     pred_data.update(pred_data_hr)
    #     pred_data.update(pred_data_rr)
    #
    #     ref_data.update(ref_data_hr)
    #     ref_data.update(ref_data_rr)
    #
    #     plot_preds_refs(pred_data, ref_data)


    ### PREDS VS REFS ###

    ### PLOT REF  ###

    args = args()
    data = load_data(args, log=[])
    plot(args.sensor_type, data)

    ### PLOT REF  ###

