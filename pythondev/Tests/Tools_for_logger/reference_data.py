# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential
# This script is used for the dblogger to show the reference data to the FAE

import argparse
import math

from Tests.Utils.DBUtils import *
from Tests.SessionHandler import _load_gps, _load_spo2
from Tests.Plots.PlotRawDataRadarCPX import load_data_from_tlog
from Tests.SessionHandler import _load


def args() -> argparse.Namespace:
    """ Parse arguments"""
    parser = argparse.ArgumentParser(description='load data for session, vital sign')
    parser.add_argument('-data_type', metavar='data_type', type=str, help='hr / rr / hrv / ra', required=False,
                        default=None)
    parser.add_argument('-path_to_file', metavar='path', type=str, help='Path to file', required=True, default=None)
    parser.add_argument('-sensor_type', metavar='sensor', type=str, help='bio / spo2 / ecg / bitalino', required=True,
                        default=None)
    parser.add_argument('-dist', type=float, required=False, help='distance to subject for fm_cw', default=None)

    return parser.parse_args()


def _load_mhe(data_path: str) -> np.ndarray:
    data = _load_spo2(data_path)
    return data[:, 1][::54]


def _load_gsensor(data_path: str) -> np.ndarray:
    data = np.loadtxt(data_path, delimiter='\t', skiprows=2, dtype=np.str)[::101, 0:3].astype(float)
    return [(raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2) ** 0.5 for raw in data]


def load_data_type(path_to_file, sensor_type, data_type, data=None):
    if data_type in ['occupancy', 'zrr', 'rest', 'stationary', 'speaking'] and \
            sensor_type in ['GT_OCCUPANCY', 'GT_ZRR', 'GT_REST', 'GT_STATIONARY', 'GT_SPEAKING']:
        formatted_data = load_ref(path=path_to_file, sensor_type=ref_sensor_type(sensor_type),
                        vital_sign_type=VitalSignType('stat')).astype('int')
    else:
        formatted_data = load_ref(path=path_to_file, sensor_type=ref_sensor_type(sensor_type),
                        vital_sign_type=VitalSignType(data_type), data=data)
        if (sensor_type == 'BIOPAC' and data_type == 'hri') or \
                (sensor_type == 'ZEPHYR_ECG' and data_type == 'hri'):
            formatted_data = np.ediff1d(formatted_data)
    return formatted_data


def filter_data(data, path_to_file, sensor_type, data_type, log):
    vs = {data_type: {'data': data}}
    try:
        if 'NES' not in sensor_type:
            unfiltered = vs[data_type]['data']
            filtered = [x for x in unfiltered if not math.isnan(x.real) or x is not None]
            filtered = [x for x in filtered if x.real >= 0]
        else:
            unfiltered = vs[data_type]['data']
            filtered = vs[data_type]['data']
        if not filtered:
            message = 'the ref data is empty. path_to_file: {}, sensor_type: {}, data_type: {}'.format(
                path_to_file, sensor_type, data_type)
            log.append(message)
            # sys.exit(1)
        return {data_type: [np.asarray(filtered), unfiltered]}
    except TypeError:
        message = 'the ref data you requested raised an error. path_to_file: {}, sensor_type: {}, ' \
                  'data_type: {}'.format(path_to_file, sensor_type, data_type)
        log.append(message)
        sys.exit(1)


def load_data_from_csv(path_to_file, data_type):
    formatted_data = {}
    with open(path_to_file) as ff:
        first_line = ff.readline()
        num_of_columns = len(first_line.split(',')) - 1
    pred = np.recfromcsv(path_to_file, usecols=list(range(num_of_columns)), encoding='UTF-8')
    formatted_data[data_type] = pred[data_type]
    if data_type in ['hr', 'rr']:
        formatted_data[data_type + '_cvg'] = pred[data_type + '_cvg']
    return formatted_data


def load_data(args, log):
    if args.sensor_type in ['GPS', 'GSENSOR', 'MHE']:
        if args.sensor_type == 'GPS':
            formatted_data = _load_gps(args.path_to_file)
        elif args.sensor_type == 'GSENSOR' and args.data_type == 'position':
            formatted_data = _load_gsensor(args.path_to_file)
        elif args.sensor_type == 'MHE' and args.data_type == 'hr':
            formatted_data = _load_mhe(args.path_to_file)
    else:
        if 'NES' in args.sensor_type:
            if args.path_to_file.endswith('.csv'):
                formatted_data = load_data_from_csv(args.path_to_file, args.data_type)
            else:
                try:
                    formatted_data = load_data_from_tlog(args.path_to_file, db, dist=args.dist)
                except ValueError:
                    formatted_data = None
                args.data_type = 'data'
        else:
            if 'data_type' not in vars(args) or args.data_type is None:
                args.data_type = []
                res_dict = {}
                data = _load(args.path_to_file, ref_sensor_type(args.sensor_type))
                for k in reference_order.keys():
                    if args.sensor_type in reference_order[k]:
                        if args.sensor_type == 'EPM_10M':
                            if k == 'bbi':
                                if 'ECG' in os.path.basename(args.path_to_file):
                                    args.data_type.append(k)
                            else:
                                if 'ECG' not in os.path.basename(args.path_to_file):
                                    args.data_type.append(k)
                        else:
                            args.data_type.append(k)
                for dtype in args.data_type:
                    formatted_data = load_data_type(args.path_to_file, args.sensor_type, dtype, data)
                    res_dict.update(filter_data(formatted_data, args.path_to_file, args.sensor_type, dtype, log))
                return res_dict
            else:
                formatted_data = load_data_type(args.path_to_file, args.sensor_type, args.data_type)
    return filter_data(formatted_data, args.path_to_file, args.sensor_type, args.data_type, log)


if __name__ == '__main__':

    args = argparse.Namespace(
        path_to_file='/Neteera/DATA/2021/5/6/4992/6647/REFERENCE/EPM_10M/6647_EPM_10M_ECG_II-20210506101559.csv',
        sensor_type='EPM_10M', data_type='bbi', dist=1200)
    data = load_data(args, log=[])
    print(data)
    # cmds = [
    # ' -path_to_file D:\2019\12\11\2278\2449\REFERENCE\IMOTION_BIOPAC\2449_IMOTION_BIOPAC_NS1193_Simulator_Processed_v2.xlsx -sensor_type IMOTION_BIOPAC'
    # ' -path_to_file D:\2019\12\11\2278\2449\REFERENCE\IMOTION_BIOPAC\2449_IMOTION_BIOPAC_NS1193_Simulator_Processed_v2.xlsx -sensor_type IMOTION_BIOPAC -data_type hr',
    # ' -path_to_file D:\2019\12\11\2278\2449\REFERENCE\IMOTION_BIOPAC\2449_IMOTION_BIOPAC_NS1193_Simulator_Processed_v2.xlsx -sensor_type IMOTION_BIOPAC -data_type rr',
    # ' -path_to_file D:\2020\4\19\1935\2099\REFERENCE\GE_B40\2099_GE_B40_ParameterData-20200419115240.csv -sensor_type EPM_10M -data_type hr',
    # ' -path_to_file D:\2020\4\19\1935\2099\REFERENCE\GE_B40\2099_GE_B40_ParameterData-20200419115240.csv -sensor_type EPM_10M -data_type rr',
    # ' -path_to_file D:\2020\5\20\2445\2616\REFERENCE\CAPNOGRAPH_PC900B\2616_CAPNOGRAPH_PC900B_CAPNOGRAPH_PC900B.xlsx -sensor_type CAPNOGRAPH_PC900B -data_type hr',
    # ' -path_to_file D:\2020\5\20\2445\2616\REFERENCE\CAPNOGRAPH_PC900B\2616_CAPNOGRAPH_PC900B_CAPNOGRAPH_PC900B.xlsx -sensor_type CAPNOGRAPH_PC900B -data_type rr',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_BW\\1726_ZEPHYR_BW_BR_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_BW -data_type ra',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hri',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_sdnn',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_nni_mean',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_nni_cv',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_lf_max_power',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_hf_max_power',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_ECG\\1726_ZEPHYR_ECG_ECG_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_ECG -data_type hrv_lf_auc',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_HR_RR\\1726_ZEPHYR_HR_RR_Summary_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_HR_RR -data_type hr',
    # ' -path_to_file D:\\2020\\3\\23\\1564\\1726\REFERENCE\ZEPHYR_HR_RR\\1726_ZEPHYR_HR_RR_Summary_Data_2020_03_23-10_18_06.csv -sensor_type ZEPHYR_HR_RR -data_type rr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hri',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_sdnn',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_nni_mean',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_nni_cv',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_lf_max_power',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_hf_max_power',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_lf_auc',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hrv_hf_auc',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type hr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type rr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type ra',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1182\\1182_BIO.txt -sensor_type BIOPAC -data_type inhale_exhale',
    # ' -path_to_file "N:\\yuval.shifriss\\Research\\DB\\a.txt" -sensor_type BITALINO -data_type hr',
    # ' -path_to_file N:\\yuval.shifriss\\Research\\DB\\spo2.csv -sensor_type ECG_SPO2 -data_type hr',
    # ' -path_to_file N:\\yuval.shifriss\\Research\\DB\\spo2_bad.txt -sensor_type SPO2 -data_type  hr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\North_Well_Integration\\Night_test_6.3.19\\small_file.edf -sensor_type NATUS -data_type hr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\1697\\190611_1042_2600100000340015.csv -sensor_type DN_ECG -data_type hri',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\ADI\\Johannes_2.July\\179526_hexoskin.txt -sensor_type HEXOSKIN -data_type hr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\ADI\\Johannes_2.July\\179526_hexoskin.txt -sensor_type HEXOSKIN -data_type rr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\353\\353_NES.txt -sensor_type NES_IR -data_type data',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\984\\984_SR.txt -sensor_type NES_SR -data_type data',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\2100\\2100_NES.tlog -sensor_type NES_SR_CW -data_type data',
    # ' -path_to_file D:\\2020\\1\\31\\700\\810\\REFERENCE\\GT_CS\\810_GT_CS_ground_truth_31_Jan_2020_150212_cs.npy -sensor_type GT_OCCUPANCY -data_type occupancy',
    # ' -path_to_file D:\\2020\\1\\31\\700\\810\\REFERENCE\\GT_ZRR\\810_GT_ZRR_ground_truth_31_Jan_2020_150212_zrr.npy -sensor_type GT_ZRR -data_type zrr',
    # ' -path_to_file D:\\2020\\1\\31\\700\810\REFERENCE\GT_REST\\810_GT_REST_ground_truth_31_Jan_2020_150212_rest.npy -sensor_type GT_REST -data_type rest'
    # ' -path_to_file D:\\2020\\1\\31\\700\810\REFERENCE\GT_STATIONARY\810_GT_STATIONARY_ground_truth_31_Jan_2020_150212_stationary.npy -sensor_type GT_STATIONARY -data_type stationary',
    # ' -path_to_file D:\\2020\\1\\31\\700\810\REFERENCE\GT_SPEAKING\810_GT_SPEAKING_ground_truth_31_Jan_2020_150212_speaking.npy -sensor_type GT_SPEAKING -data_type speaking'
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\343\\343_GSENSOR.txt -sensor_type GSENSOR -data_type position',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\343\\343_GPS.txt -sensor_type GPS -data_type velocity',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\855\\log_2018_10_19_10_18_04_JY_engine_off_Oximeter.txt -sensor_type MHE -data_type hr',
    # ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\87\\87_Wave_II.csv -sensor_type ECG_WAVE_II -data_type hr'
    # ' -path_to_file D:\\2020\\4\\12\\1845\\2009\\REFERENCE\\ELBIT_ECG_SAMPLE\\2009_ELBIT_ECG_SAMPLE_ecg_15866725825957507.csv -sensor_type ELBIT_ECG_SAMPLE -data_type hr'

    # NOT SUPPORTED:
    #     ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\353\\353_NES.txt -sensor_type NES_IR -data_type duration',
    #     ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\984/984_SR.txt -sensor_type NES_SR -data_type duration',
    #     ' -path_to_file N:\\NeteeraVirtualServer\\Data\\Tests\\V3_TESTS\\2100/2100_NES.tlog -sensor_type NES_SR_CW -data_type duration',


