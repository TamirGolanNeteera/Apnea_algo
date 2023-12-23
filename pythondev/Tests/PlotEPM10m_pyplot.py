# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))    # noqa

from Tests.vsms_db_api import DB, Sensor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import argparse
from os import path
from scipy import signal
#import warnings
#warnings.filterwarnings("ignore")


# constants
ECG_PARAMS = {'height': 100, 'distance': 0.7*250, 'denom': 500}
CO2_PARAMS = {'height': 3600, 'distance': 55, 'denom': 40}
PARAMS = {'ECG': ECG_PARAMS, 'CO2': CO2_PARAMS}


def vector_pad(vector, other_vector):
    """ Takes a vector, another vector, and adds the last value of the first vector
    to the end of it, as many times as needed so it will be in the length of the other vector"""
    remainder = other_vector.shape[0] - vector.shape[0]
    right_pad = np.full(remainder, vector[-1])
    return np.concatenate((vector, right_pad))


def extract_values_from_csv(csv_pth: str) -> (np.ndarray, np.ndarray):
    """
    @param csv_pth: Path to csv file
    @type csv_pth: str
    @return: The x values and y values of the graph
    @rtype: tuple(NumPy array, NumPy array)
    """
    df = pd.read_csv(csv_pth, header=None)

    n = len(df.columns)
    new_header = ['second'] + list(range(1, n))
    df.columns = new_header
    epm_values = df.values.T[1:].T
    epm_values = epm_values.reshape(epm_values.shape[0] * epm_values.shape[1], )
    x_values = np.ndarray(epm_values.shape[0]) / len(df.columns)
    return x_values, epm_values


class Graph:

    def __init__(self, name: str, x: np.ndarray, y: np.ndarray):
        self.name = name
        self.x = x
        self.y = y
        self.peaks = self.get_peaks()

    def get_peaks(self):
        """
        @return: The peaks in the graph
        @rtype: tuple[np.ndarray, np.ndarray]
        """
        d = PARAMS[self.name]
        if self.name == 'ECG' and d['denom'] == 500:
            d['height'] = np.mean(self.y) + 2.5 * np.std(self.y)
            peaks_up, peaks_up_y = signal.find_peaks(self.y, height=d['height'], distance=d['distance'])
            peaks_down, peaks_down_y = signal.find_peaks(-1*self.y, height=d['height'], distance=d['distance'])
            avg_dist_up = np.mean(np.abs(peaks_up_y['peak_heights'] - np.mean(self.y)))
            avg_dist_down = np.mean(np.abs(peaks_down_y['peak_heights'] - np.mean(self.y)))
            avg_dist_up = 0 if len(peaks_up) == 0 else avg_dist_up
            avg_dist_down = 0 if len(peaks_down) == 0 else avg_dist_down
            peaks = peaks_up if avg_dist_up >= avg_dist_down else peaks_down
            self.y = self.y if avg_dist_up >= avg_dist_down else -1*self.y
        else:
            y_mean = np.mean(self.y)
            d['height'] = y_mean + 0.6*np.std(self.y)
            diff_signed_y = np.diff(np.sign(self.y - d['height']))
            up_z_cross = np.where(diff_signed_y > 0)[0]
            down_z_cross = np.where(diff_signed_y < 0)[0]
            rr_peaks = []
            if up_z_cross[0] > down_z_cross[0]:
                up_z_cross = np.concatenate((np.zeros(1).astype(int), up_z_cross))
                down_z_cross = np.concatenate((down_z_cross, np.array([self.y.shape[0]]).astype(int)))
            for u, dn in zip(up_z_cross, down_z_cross):
                delta = self.y[u:dn]
                peak_loc = np.argmax(delta) + u
                if len(rr_peaks) > 0 and peak_loc - rr_peaks[-1] < d['denom'] / 2:
                    continue
                rr_peaks.append(peak_loc)
            peaks = np.array(rr_peaks)
        return peaks, self.y[peaks]

    def compute_rate(self):
        """
        @return: The rate of peaks per minute for each second
        @rtype: NumPy array
        """
        denom = PARAMS[self.name]['denom']
        peaks_x, _ = self.peaks
        rr_list = np.diff(peaks_x) * (1000 / denom)  # convert to milliseconds
        bpm = 60000 / rr_list
        filler = 60 if self.name == 'ECG' else 12
        n_seconds_back = 12 if self.name == 'ECG' else 60
        sparse_bpm = np.full((self.x.shape[0], ), filler)

        # spread the data over the time axis
        for i, x in enumerate(peaks_x[:-1]):
            x_p1 = peaks_x[i+1]
            sparse_bpm[x:x_p1] = bpm[i]

        # make average over window in size second
        seconds_bpm = []
        for i in range(0, sparse_bpm.shape[0] - denom, denom):
            seconds_bpm.append(np.mean(sparse_bpm[i:i+denom]))

        # moving average of n seconds
        avg_n_sec = []
        for i in range(len(seconds_bpm)):
            to_compute = np.mean(seconds_bpm[i - n_seconds_back:i]) if i > n_seconds_back \
                else np.mean(seconds_bpm[0:i + 1])
            avg_n_sec.append(to_compute)
        return np.array(avg_n_sec)


class Session:

    def __init__(self, input_path, setup_id):
        self.graphs = []
        self.db = DB()
        input_path = self.db.setup_dir(setup_id)
        input_path = input_path[:input_path.rfind('/')]

        self.db.update_mysql_db(int(setup_id))
        co2_path, ecg_path, parameters_path = gen_epm_paths(input_path, setup_id, self.db)
        self.setup_num = setup_id#co2_path.split('/')[-1][:len(str(setup_id))]
        co2_x, co2_y = extract_values_from_csv(co2_path)
        self.graphs.append(Graph('CO2', co2_x, co2_y))
        if ecg_path is not None:
            ecg_x, ecg_y = extract_values_from_csv(ecg_path)
            self.graphs.append(Graph('ECG', ecg_x, ecg_y))
        self.pmd_path = parameters_path

    def plot_session(self, savefig=False, out_path=None):
        """
        Plots the session using pyplot
        """
        lines = []
        scatters = []
        df = pd.read_csv(self.pmd_path)

        for graph in self.graphs:
            epm_values = graph.y.astype(int)
            var = graph.name
            denom = PARAMS[graph.name]['denom']
            time_stamp = np.arange(epm_values.shape[0]) / denom
            bpm = graph.compute_rate()
            filler = np.zeros(int(graph.x.shape[0] // denom) + 1)
            bpm = vector_pad(bpm, filler)

            if var == 'CO2':
                fig, ax = plt.subplots()
                fig.set_figheight(8)
                fig.set_figwidth(15)
                plt.rc('grid', linestyle='--', color='black')
                plt.grid()
                plt.xlabel('Seconds')
                plt.xlim(xmin=0, xmax=epm_values.shape[0] / denom)

                line_plot, = ax.plot(time_stamp, epm_values, label=var, color='green')
                peaks_x, peaks_y = graph.peaks
                peaks_x = peaks_x.astype(float) / denom
                scatter_plot = ax.scatter(peaks_x, peaks_y, color='red', label='Peaks')
                ax.legend(loc='lower left')
                ax.set_ylabel('CO2 millivolts', color='tab:green')
                print('CO2 Num peaks: ' + str(len(peaks_x)) +
                      '\nAverage peaks per minute on last 60 sec.: {:.2f}\n'.format(np.median(bpm[-61:])))
                ax_b = ax.twinx()
                ax_b.set_ylabel('Respirations Per Minute', color='tab:purple')
                ax_b.plot(np.arange(bpm.shape[0]), bpm, color='indigo', label='Our calculations')
                rr_values = df[' RR(rpm)'].to_numpy()
                ax_b.plot(np.arange(len(rr_values)), rr_values, color='crimson', label='Device calculation')
                ax_b.set_ylim([0, 40])
                plt.title('')#'CO2 graph for setup ' + str(self.setup_num) + ' ' + self.db.setup_sn(self.setup_num)[0:3] + ' ' + self.db.setup_posture(self.setup_num) + ' ' + str(self.db.setup_distance(self.setup_num)))
                ax_b.legend(loc=4)

            else:

                if denom != 500:
                    graph.name = 'Pleth'
                fig, ax2 = plt.subplots()
                fig.set_figheight(8)
                fig.set_figwidth(15)
                plt.rc('grid', linestyle='--', color='black')
                plt.grid()
                plt.xlabel('Seconds')
                plt.xlim(xmin=0, xmax=epm_values.shape[0] / denom)

                line_plot, = ax2.plot(time_stamp, epm_values, label=graph.name)
                peaks_x, peaks_y = graph.peaks
                peaks_x = peaks_x.astype(float) / denom
                scatter_plot = ax2.scatter(peaks_x, peaks_y, color='red', label='Peaks')
                ax2.set_ylabel(graph.name + ' millivolts', color='tab:blue')
                plt.xlim()
                print(graph.name + ' Num peaks: ' + str(len(peaks_x)) +
                      '\nAverage peaks per minute on last 60 sec.: {:.2f}'.format(np.median(bpm[-61:])))
                ax2_b = ax2.twinx()
                ax2_b.set_ylabel('Beats Per Minute', color='tab:red')
                ax2_b.plot(np.arange(bpm.shape[0]), bpm, color='maroon', label='Our calculation')
                hr_values = df[' HR(bpm)'].to_numpy()
                ax2_b.plot(np.arange(len(hr_values)), hr_values, color='darkorange', label='Device calculation')
                ax2_b.set_ylim([40, 160])
                ax2.legend(loc='lower left')

                ax2.legend()
                ax2_b.legend(loc=4)
                plt.title(f'{graph.name} graph for setups {self.db.setup_multi(self.setup_num)}')

            lines.append(line_plot)
            scatters.append(scatter_plot)
        #
        # if PARAMS['ECG']['denom'] != 500:
        #     popup_window = tk.Tk()
        #     popup_window.wm_title("Attention")
        #     popup_window.geometry("250x120")
        #     popup_window.eval('tk::PlaceWindow . center')
        #     pop = tk.Text(popup_window)
        #     pop.insert(tk.END, 'No ECG raw data.\n Plotting Pleth raw data instead.')
        #     pop.grid(row=0, column=1)
        #     b = tk.Button(popup_window, text='OK', command=popup_window.destroy)
        #     b.place(relx=0.5, rely=0.8, anchor=tk.S)
        #     popup_window.attributes("-topmost", True)

        if savefig:
            plt.savefig(str(self.setup_num)+'_co2.png')
        else:
            plt.show()


def gen_epm_paths(dir_path, setup, db):
    """
    Generates the csv paths to files we need for plotting the graph.
    Only one of the arguments is not None.
    @param dir_path: Path to EPM_10M directory
    @type dir_path: str
    @param setup: The setup id number
    @type setup: str or int
    @return: The three paths: ECG, CO2 raw data and Parameter Data file
    @rtype: tuple(str, str, str)
    @param db: database handler
    """
    if dir_path is not None:
        my_path = dir_path
        setup = my_path.split('/')[6]
        db.update_mysql_db(setup)
    else:
        my_path = db.setup_dir(setup)
        dir_path = my_path[:my_path.rfind('/')]
        print(my_path)
        print("...")
    setups = os.listdir(dir_path)
    setups.sort()
    ref_path = os.path.join(os.path.join(os.path.join(dir_path, setups[0]), 'REFERENCE'), 'EPM_10M')
    epm_files = os.listdir(ref_path)
    full_path_ecgs = [f for f in epm_files if 'ECG' in f and '.csv' in f]
    if len(full_path_ecgs) == 0:
        full_path_ecg =  [f for f in epm_files if 'Pleth' in f ][0]
        if path.getsize(full_path_ecg) == 0:
            full_path_ecg = None
        ECG_PARAMS['denom'], ECG_PARAMS['height'] = 60, 60
    else:
        full_path_ecg = full_path_ecgs[0]
    full_path_co2 = [f for f in epm_files if 'CO2' in f and 'csv' in f][0]
    full_path_pmd = [f for f in epm_files if 'ParameterData' in f and 'csv' in f][0]
    return os.path.join(ref_path, full_path_co2), os.path.join(ref_path,full_path_ecg), os.path.join(ref_path,full_path_pmd)

def gen_epm_paths2(dir_path, setup, db):
    """
    Generates the csv paths to files we need for plotting the graph.
    Only one of the arguments is not None.
    @param dir_path: Path to EPM_10M directory
    @type dir_path: str
    @param setup: The setup id number
    @type setup: str or int
    @return: The three paths: ECG, CO2 raw data and Parameter Data file
    @rtype: tuple(str, str, str)
    @param db: database handler
    """
    if dir_path is not None:
        my_path = dir_path
        setup = my_path.split('/')[6]
        db.update_mysql_db(setup)


    full_path_ecgs = [f for f in db.setup_ref_path(setup=setup, sensor=Sensor.epm_10m, search='ECG')
                      if path.getsize(f) > 0]
    if len(full_path_ecgs) == 0:
        full_path_ecg = db.setup_ref_path(setup=setup, sensor=Sensor.epm_10m, search='Pleth')[0]
        if path.getsize(full_path_ecg) == 0:
            full_path_ecg = None
        ECG_PARAMS['denom'], ECG_PARAMS['height'] = 60, 60
    else:
        full_path_ecg = full_path_ecgs[0]
    full_path_co2 = db.setup_ref_path(setup=setup, sensor=Sensor.epm_10m, search='CO2')[0]
    full_path_pmd = db.setup_ref_path(setup=setup, sensor=Sensor.epm_10m)[0]
    return full_path_co2, full_path_ecg, full_path_pmd
def one_hot_peaks(peaks_x: np.ndarray, vec_size: int) -> np.ndarray:
    """
    @param peaks_x: Peaks x values
    @type peaks_x: NumPy Array
    @param vec_size: The original data vector size
    @type vec_size: int
    @return: Vector with ones wherever there is a peak and zeros else
    @rtype: NumPy Array
    """
    one_hot = np.zeros(vec_size)
    one_hot[peaks_x] = 1
    return one_hot


def plot_parameter_data_and_moving_peaks(pmd_pth, bpm, rr=False):
    """
    Plots the given graph (HR or RR) and reference using pyplot
    @param pmd_pth: Path to parameter data csv
    @type pmd_pth: str
    @param bpm: Beats per minute vector of seconds
    @type bpm: NumPy array
    @param rr: Flag if it is RR or not (HR)
    @type rr: Boolean
    """
    df = pd.read_csv(pmd_pth)
    fig_num = 1 if rr else 2
    curr_fig = plt.figure(fig_num)
    curr_fig.set_figheight(8)
    curr_fig.set_figwidth(15)
    plt.rc('grid', linestyle='--', color='black')
    plt.grid()

    if rr:
        plt.title('Setup number ' + pmd_pth.split('/')[6] + ' RR calculation, compared to EPM-10m\'s result')
        plt.plot(np.arange(len(bpm)), bpm, label='Neteera RR calculation')
        rr_values = df[' RR(rpm)'].to_numpy()
        plt.plot(np.arange(len(rr_values)), rr_values, label='RR reference', alpha=0.65)
        plt.xlabel('Seconds')
        plt.ylabel('RPM')
    else:
        plt.title('Setup number ' + pmd_pth.split('/')[6] + ' BPM calculation, compared to EPM-10m\'s result')
        plt.plot(np.arange(len(bpm)), bpm, label='Neteera BPM calculation')
        hr_values = df[' HR(bpm)'].to_numpy()
        plt.plot(np.arange(len(hr_values)), hr_values, label='HR reference', alpha=0.65)
        plt.xlabel('Seconds')
        plt.ylabel('BPM')
    plt.legend()
#
#
# def tk_logic():
#     """
#     runs the gui logic if path not given
#     means - asks the user for setup id,
#     and gets data and plots it accordingly
#     """
#     master = tk.Tk()
#     master.eval('tk::PlaceWindow . center')
#     master.geometry("200x120")
#     master.title("PlotEPM10m")
#     master.wm_attributes('-topmost', 1)
#     tk.Label(master, text="Setup id").place(relx=0.5, rely=0.15, anchor=tk.S)
#     e1 = tk.Entry(master)
#     e1.place(relx=0.5, rely=0.3, anchor=tk.S)
#     e2 = tk.Entry(master)
#     e2.place(relx=0.5, rely=0.75, anchor=tk.S)
#     root = tk.Tk()
#     root.withdraw()
#     bool_flag = False
#     tk.Label(master, bg='white', width=20, text='')
#     button = tk.Button(master, text='Go', command=master.quit)
#     button.place(relx=0.5, rely=1, anchor=tk.S)
#     hr_check = tk.IntVar(value=0)
#     rr_check = tk.IntVar(value=0)
#     c1 = tk.Checkbutton(e2, text='HR(bpm)', variable=hr_check, onvalue=1, offvalue=0)
#     c1.pack()
#     c2 = tk.Checkbutton(e2, text='RR(rpm)', variable=rr_check, onvalue=1, offvalue=0)
#     c2.pack()
#
#     while not bool_flag:
#         tk.mainloop()
#         setup_number = e1.get()
#         if setup_number.isdigit() and len(setup_number) > 0:
#             session = Session(None, setup_number)
#             if rr_check.get():
#                 co2_bpm = session.graphs[0].compute_rate()
#                 plot_parameter_data_and_moving_peaks(session.pmd_path, co2_bpm, True)
#             if hr_check.get():
#                 ecg_bpm = session.graphs[1].compute_rate()
#                 plot_parameter_data_and_moving_peaks(session.pmd_path, ecg_bpm)
#             session.plot_session()
#             bool_flag = True
#     master.destroy()


def parse_args():
    """
    @return: dictionary of parameter values
    @rtype: dictionary
    """
    parser = argparse.ArgumentParser(description='Get arguments to plot')
    parser.add_argument('-path', metavar='file_path', type=str,
                        help='The location of the file on the server', required=False)
    parser.add_argument('-setup_id', metavar='setup_num', type=str,
                        help='The id of the setup', required=False)
    parser.add_argument('--hr', action='store_true', help='Weather to plot HR estimation graph', required=False)
    parser.add_argument('--rr', action='store_true', help='Weather to plot RR estimation graph', required=False)
    parser.add_argument('-save_path', metavar='save_path', type=str,
                        help='The location of where to save plots', required=False)
    return parser.parse_args()


def work_with_args(args):
    """
    Plots the wanted graphs given the args
    @param args: Argument values dictionary
    @type args: arg_parse parser
    """
    setups = [10604]

    for s in setups:
        session = Session(None, s)
        if args.rr:
            co2_bpm = session.graphs[0].compute_rate()
            plot_parameter_data_and_moving_peaks(session.pmd_path, co2_bpm, True)
        if args.hr:
            ecg_bpm = session.graphs[1].compute_rate()
            plot_parameter_data_and_moving_peaks(session.pmd_path, ecg_bpm)
        session.plot_session(savefig=False, out_path=args.save_path)


if __name__ == '__main__':
    k = sys.argv
    if len(sys.argv) > 1:
        arguments = parse_args()
        work_with_args(arguments)
    # else:
    #     tk_logic()
