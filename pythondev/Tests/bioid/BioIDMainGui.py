
import Spot
import bioid.BioidML as bml

from Tests.Utils.LoadingAPI import load_nes
from Tests.vsms_db_api import DB
from BioIDUtils import *

from pylibneteera.datatypes import VitalSignType

import warnings
import sys
import datetime
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
import argparse
from PIL import ImageTk, Image
import json
import pickle
import pathlib
import re
from operator import itemgetter
import numpy as np
import os


# In the future may automatically run batch file
# import setupHandler
# subprocess.call([r'/Neteera/Work/homes/eldar.hacohen/Logger3.5.18.20/runFMCW.bat'])

warnings.filterwarnings("ignore", category=DeprecationWarning)
NUM_ROWS = 15000


class MainMenu:
    # The main menu GUI

    def __init__(self):
        self.master = tk.Tk()
        self.master.eval('tk::PlaceWindow . center')
        self.master.geometry("360x200")
        self.master.title("BioID")
        self.master.wm_attributes('-topmost', 1)
        tk.Label(self.master, text="Welcome to BioID tool").place(relx=0.5, rely=0.2, anchor=tk.S)
        online_button = tk.Button(text="Run Online", command=self.create_new_online)
        online_button.place(relx=0.2, rely=0.5, anchor=tk.S)
        offline_button = tk.Button(text="Run offline", command=self.create_new_offline)
        offline_button.place(relx=0.8, rely=0.5, anchor=tk.S)
        add_id_button = tk.Button(text="Add new subject", command=self.create_new_add_subj)
        add_id_button.place(relx=0.2, rely=0.7, anchor=tk.S)
        remove_id_button = tk.Button(text="Remove subject", command=self.create_new_remove_subj)
        remove_id_button.place(relx=0.8, rely=0.7, anchor=tk.S)
        tk.mainloop()

    def create_new_online(self):
        self.master.quit()
        self.master.destroy()
        OnlineWindow()

    def create_new_offline(self):
        self.master.quit()
        self.master.destroy()
        OfflineWindow()

    def create_new_add_subj(self):
        self.master.quit()
        self.master.destroy()
        AddSubject()

    def create_new_remove_subj(self):
        self.master.quit()
        self.master.destroy()
        RemoveSubject()


class Window:
    # A father class for online and offline windows

    def __init__(self):
        self.db = DB()
        self.args = argparse.Namespace()
        self.master = tk.Tk()
        self.master.eval('tk::PlaceWindow . center')
        self.master.wm_attributes('-topmost', 1)
        self.identity = ''
        self.percentage = ''
        self.back_button = tk.Button(text="back to menu", command=self.back_to_menu)
        self.back_button.place(relx=0.5, rely=0.6, anchor=tk.S)
        self.label = None

    def back_to_menu(self):
        self.master.quit()
        self.master.destroy()
        menu = MainMenu()

    def set_prediction(self):
        """
        Shows the prediction of the NN to the window
        @return:
        """
        text_box = tk.Text(self.master, height=3, width=30)
        text_box.place(relx=0.5, rely=0.85, anchor=tk.S)
        # back_button = tk.Button(text="back to menu", command=self.back_to_menu)
        self.back_button.place(relx=0.5, rely=0.95, anchor=tk.S)

        threshold = 30
        if self.percentage == '':
            self.percentage = '0'
        if float(self.percentage) < threshold:
            self.identity = 'Unknown'
        text_box.insert(tk.END, 'Prediction: ' + self.identity + '\nConfidence Level: ' + self.percentage + '%')
        if self.identity == 'loading...':
            self.identity = 'Unknown'
        out_loc = os.path.dirname(sys.argv[0]).split('/Tests')[0]
        try:
            img = ImageTk.PhotoImage(Image.open(out_loc + '/bioid/resources/pictures/' + self.identity + '.png'))
        except Exception:
            img = ImageTk.PhotoImage(Image.open(out_loc + '/bioid/resources/pictures/Unknown.png'))
        if self.label is not None:
            self.label.destroy()
        self.label = tk.Label(self.master)
        self.label.image = img
        self.label.configure(image=img)
        self.label.pack(side=tk.TOP)


class OnlineWindow(Window):
    # Class for window that shows online prediction
    def __init__(self):
        super().__init__()
        self.master.geometry("400x200")
        self.master.title("Online BioID")
        self.identity = ''
        self.percentage = ''
        self.sess = 0
        self.label = None
        dirbutton = tk.Button(text="choose directory", command=self.choose_dir)
        dirbutton.place(relx=0.5, rely=0.8, anchor=tk.S)
        button3 = tk.Button(text="Get Prediction", command=self.master.quit)
        button3.place(relx=0.5, rely=0.4, anchor=tk.S)
        self.i = 49
        self.res_dict = {}
        self.dir_name = ''

        tk.mainloop()

        self.master.geometry("400x600")
        self.get_prediction()

    def choose_dir(self):
        self.dir_name = fd.askdirectory()

    def update_prediction(self):
        """
        Updates the prediction every 10 seconds. Runs until user quits
        @return:
        """
        model_path = os.path.join(os.getcwd(), 'bioid', 'resources', 'nn_classifier')
        ttlog_paths = {f: datetime.datetime.fromtimestamp(pathlib.Path(os.path.join(
            self.dir_name, f)).stat().st_mtime) for
                       f in os.listdir(self.dir_name) if f.endswith('.ttlog')}
        ttlog_path = sorted(ttlog_paths.items(), key=lambda x: x[1], reverse=True)[0][0]
        filename = os.path.join(self.dir_name, ttlog_path)
        data, ended = read_last_30_sec(filename)
        if -1 in data.keys():
            self.identity = 'loading...'
            self.percentage = '100'
        else:
            data.update(get_dict_values(filename))
            r = Spot.prepare_data(data, 500.0)
            self.res_dict[VitalSignType.identity] = bml.compute_identity(
                r,
                classifier='nn',  # 'nn' or 'linear'
                model_path=model_path)
            self.i += NUM_ROWS // 3
            self.identity = self.res_dict[VitalSignType.identity]['identity']
            self.percentage = str(np.round(self.res_dict[VitalSignType.identity]['class_prevalence'] * 100, 2))
        self.set_prediction()
        self.master.update_idletasks()
        self.master.after(10000, self.update_prediction)

    def get_prediction(self):
        """
        Calls update_prediction and runs mainloop
        """
        self.label = tk.Label(self.master)
        self.master.after(1, self.update_prediction)
        self.master.mainloop()


class OfflineWindow(Window):
    # Class for

    def __init__(self):
        super().__init__()
        self.master.geometry("400x600")
        self.master.title("Offline BioID")
        self.sess = 0
        tk.Label(self.master, text="Setup id").place(relx=0.5, rely=0.1, anchor=tk.S)
        e1 = tk.Entry(self.master)
        e1.place(relx=0.5, rely=0.2, anchor=tk.S)
        button3 = tk.Button(text="Get Prediction", command=self.master.quit)
        button3.place(relx=0.5, rely=0.4, anchor=tk.S)
        self.res_dict = {}
        tk.mainloop()
        setup_num = e1.get()

        if setup_num.isdigit() and len(setup_num) == 4:
            # if legal setup number
            self.sess = int(setup_num)
            self.get_prediction()
            self.set_prediction()

    def get_prediction(self):
        """
        Runs NN model on setup sets the prediction
        """
        sess = int(self.sess)
        data = load_nes(sess, self.db)
        self.args.seed = 14
        self.args.compute = [VitalSignType.identity]
        self.args.spot_time_from_end = 6000
        self.args.classifier = 'nn'
        res_dict = Spot.summarize_setup(data, fs=500, args=self.args)
        self.identity = res_dict[VitalSignType.identity]['identity']
        self.percentage = str(np.round(res_dict[VitalSignType.identity]['class_prevalence'] * 100, 2))


class RemoveSubject:

    def __init__(self):
        self.master = tk.Tk()
        self.master.geometry("400x200")
        self.master.title("Remove Subject")
        self.master.eval('tk::PlaceWindow . center')
        self.master.wm_attributes('-topmost', 1)
        back_button = tk.Button(text="back to menu", command=self.back_to_menu)
        back_button.place(relx=0.5, rely=0.6, anchor=tk.S)
        self.subject_name = tk.StringVar()
        names_list = sorted([n[0] for n in metadata[0]])
        self.subject_name.set(names_list[0])
        drop = tk.OptionMenu(self.master, self.subject_name, *names_list)
        drop.place(relx=0.5, rely=0.2, anchor=tk.S)
        self.pic_filename = ''
        remove_button = tk.Button(text="remove subject", command=self.train_network_for)
        remove_button.place(relx=0.5, rely=0.4, anchor=tk.S)
        tk.mainloop()

    def back_to_menu(self):
        self.master.quit()
        self.master.destroy()
        MainMenu()

    def train_network_for(self):
        setup_2_name = metadata[4]
        additional_class_train = []
        additional_class_test = []
        names = metadata[1]
        FT_w_setups = {name: [sess for sess in setup_2_name.keys() if
                                setup_2_name[sess] == name] for name in names}
        for k,v in FT_w_setups.items():
            if not v:
                v = np.unique(FT_w_setups[k])
            if len(v) > 30:
                v = v[-30:]

            split_point = int(0.8 * len(v))
            print(k, "train", np.array(v)[:split_point])
            print(k, "test", np.array(v)[split_point:])
            additional_class_test.append(np.array(v)[split_point:])
            additional_class_train.append(np.array(v)[:split_point])
        print(len(FT_w_setups.items()), "subjects")
        print((np.hstack(additional_class_train)).shape[0], "train setups")
        print((np.hstack(additional_class_test)).shape[0], "test setups")
        additional_class_test = np.hstack(additional_class_test)
        additional_class_train = np.hstack(additional_class_train)

        additional_active_names = np.unique(list(itemgetter(*set(additional_class_train))(setup_2_name)))
        ft_model_data = NNModelData(additional_active_names, additional_class_train, additional_class_test)
        file_to_read = open(os.path.join(save_path, 'metadata.pkl'), "rb")
        metadata_dict = pickle.load(file_to_read)
        ft_model_data.model_json_fn = os.path.join(save_path, 'bioid_nn_model.json')
        ft_model_data.model_weights_fn = os.path.join(save_path, 'bioid_nn_weights.hdf5')
        ft_model_data_without_subj = remove_subject_from_nn(self.subject_name.get(), metadata_dict, ft_model_data)
        ft_model_data_without_subj.model_json_fn = save_path + "/bioid_nn_model.json"
        ft_model_data_without_subj.model_weights_fn = save_path + '/bioid_nn_weights.hdf5'
        ft_model_data_without_subj.model.save_weights(filepath=ft_model_data_without_subj.model_weights_fn)
        print("Saved base NN model")
        model_json = ft_model_data_without_subj.model.to_json()
        with open(ft_model_data_without_subj.model_json_fn, "w") as json_file:
            json_file.write(model_json)
        names2 = {v for v in ft_model_data.name_2_class.items()}

        meta_arr = [names2, ft_model_data_without_subj.active_names, ft_model_data_without_subj.train_setups,
                    ft_model_data_without_subj.test_setups, setup_2_name]
        np.save(os.path.join(save_path, fn), meta_arr, allow_pickle=True)

        print("Metadata saved to" + os.path.join(save_path, fn))
        tk.Label(self.master, text=self.subject_name.get() +
                                   " removed successfully!").place(relx=0.5, rely=0.9, anchor=tk.S)


class AddSubject:

    def __init__(self):
        self.master = tk.Tk()
        self.master.geometry("400x200")
        self.master.title("Add Subject")
        self.master.eval('tk::PlaceWindow . center')
        self.master.wm_attributes('-topmost', 1)
        back_button = tk.Button(text="back to menu", command=self.back_to_menu)
        back_button.place(relx=0.5, rely=0.8, anchor=tk.S)
        self.subject_name = tk.StringVar()
        neural_names = [n[0] for n in metadata[0]]
        all_names = [name for name in np.unique(list(metadata[4].values())).tolist()
                     if not bool(re.search(r'\d', name))]
        all_names.sort()
        names_list = [name for name in all_names if name not in neural_names]
        self.subject_name.set(names_list[0])
        drop = tk.OptionMenu(self.master, self.subject_name, *names_list)
        drop.place(relx=0.5, rely=0.2, anchor=tk.S)
        self.pic_filename = ''
        open_button = ttk.Button(self.master, text='Choose Picture', command=self.select_file)
        open_button.pack(expand=True)
        open_button.place(relx=0.5, rely=0.4, anchor=tk.S)
        back_button = tk.Button(text="add subject", command=self.train_network_for)
        back_button.place(relx=0.5, rely=0.6, anchor=tk.S)
        tk.mainloop()

    def select_file(self):
        """

        @return:
        """
        filetypes = (
            ('JPG files', '*.jpg'),
            ('PNG files', '*.png')
        )

        self.pic_filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        showinfo(
            title='Selected File',
            message=self.pic_filename
        )

    def back_to_menu(self):
        self.master.quit()
        self.master.destroy()
        MainMenu()

    def train_network_for(self):
        setup_2_name = metadata[4]
        removed_class_train = []
        removed_class_test = []
        names = metadata[1]
        FT_w_setups = {name: [sess for sess in setup_2_name.keys() if
                                setup_2_name[sess] == name] for name in names}
        for k,v in FT_w_setups.items():
            if not v:
                v = np.unique(FT_w_setups[k])
            if len(v) > 30:
                v = v[-30:]

            split_point = int(0.8 * len(v))
            print(k, "train", np.array(v)[:split_point])
            print(k, "test", np.array(v)[split_point:])
            removed_class_test.append(np.array(v)[split_point:])
            removed_class_train.append(np.array(v)[:split_point])
        print(len(FT_w_setups.items()), "subjects")
        print((np.hstack(removed_class_train)).shape[0], "train setups")
        print((np.hstack(removed_class_test)).shape[0], "test setups")
        removed_class_test = np.hstack(removed_class_test)
        removed_class_train = np.hstack(removed_class_train)

        additional_active_names = []
        for s in set(removed_class_train):
            additional_active_names.append(setup_2_name[s])

        additional_active_names = np.unique(additional_active_names).tolist()

        for s in set(removed_class_train):
            additional_active_names.append(setup_2_name[s])

        ft_model_data = NNModelData(additional_active_names, removed_class_train, removed_class_test)
        file_to_read = open(os.path.join(save_path, 'metadata.pkl'), "rb")
        metadata_dict = pickle.load(file_to_read)
        ft_model_data.model_json_fn = os.path.join(save_path, 'bioid_nn_model.json')
        ft_model_data.model_weights_fn = os.path.join(save_path, 'bioid_nn_weights.hdf5')
        ft_model_data_with_new_subj = add_subject_to_nn(self.subject_name.get(), metadata_dict, ft_model_data)
        ft_model_data_with_new_subj.model_json_fn = save_path + "/bioid_nn_model.json"
        ft_model_data_with_new_subj.model_weights_fn = save_path + '/bioid_nn_weights.hdf5'
        ft_model_data_with_new_subj.model.save_weights(filepath=ft_model_data_with_new_subj.model_weights_fn)
        print("Saved base NN model")
        model_json = ft_model_data_with_new_subj.model.to_json()
        with open(ft_model_data_with_new_subj.model_json_fn, "w") as json_file:
            json_file.write(model_json)
        names2 = {v for v in ft_model_data.name_2_class.items()}

        meta_arr = [names2, ft_model_data_with_new_subj.active_names, ft_model_data_with_new_subj.train_setups,
                    ft_model_data_with_new_subj.test_setups, setup_2_name]
        np.save(os.path.join(save_path, fn), meta_arr, allow_pickle=True)
        print("Metadata saved to" + os.path.join(save_path, fn))
        if self.pic_filename != '':
            im1 = Image.open(self.pic_filename)
            im1 = im1.resize((250, 320), Image.ANTIALIAS)
            im1.save(os.path.join(save_path.split('nn_classifier')[0] +
                                        'pictures', self.subject_name.get() + '.png'))
        tk.Label(self.master, text=self.subject_name.get() + " added successfully!").place(relx=0.5, rely=0.9, anchor=tk.S)


def write_next_30_seconds(in_path, out_path, row_num):
    with open(in_path, 'r') as in_f:
        lines = in_f.readlines()[row_num:row_num + NUM_ROWS]
        with open(out_path, 'w') as out_f:
            for line in lines:
                if 'Log Summary' in line:
                    return False
                out_f.write(line)
            out_f.close()
        in_f.close()
    return True


def read_last_30_sec(ttlog_path):
    count = 0
    ended = False
    with open(ttlog_path, 'r') as f:
        file_lines = f.readlines()
        for line in file_lines:

            if 'Log Summary' in line:
                ended = True
                break
            count += 1
        file_lines = file_lines[:count - 1]
        if len(file_lines) < NUM_ROWS:
            return {-1: -1}, False
        relevant = [line.split('CPX:') for line in file_lines[-NUM_ROWS:]]
        relevant = [np.array(d[1].split(',')[:-1]).astype(np.int64) for d in relevant]
        I_values, Q_values = np.array([np.array(s[::2].tolist()) for s in relevant]),  \
                             np.array([np.array(s[1::2].tolist()) for s in relevant])
        ttlog_matrix = np.array([i + 1j*q for (i, q) in zip(I_values, Q_values)])
        ttlog_dict = {'data': ttlog_matrix}
    f.close()
    return ttlog_dict, ended


def get_value(line, key):
    splitted_line = line.split(',')
    for keyval in splitted_line:
        if key in keyval:
            return int(keyval.split('=')[1])


def get_dict_values(ttlog_path):
    names_dict = {'BW': 'Bandwidth[MHz]',
    'sample_rate': 'constFR',
    'PLLConfig_PreBuf': 'PreBuf[%]',
    'PLLConfig_PostBuf': 'PostBuf[%]',
    'basebandConfig_ADC_samplesNum':'ADC_samplesNum',
    'basebandConfig_FFT_size': 'FFT_size',
    'bins_num': 'logged bins#',
    'dist': 'measuredTargetDistance'}
    values_dict = {}
    with open(ttlog_path, 'r') as f:
        for line in f:
            for k, v in names_dict.items():
                if v in line:
                    values_dict[k] = get_value(line, v)

    values_dict['bins'] = np.arange(values_dict['bins_num']) + 1
    return values_dict


if __name__ == '__main__':
    # file_path = ''
    # write_next_30_seconds('/Neteera/Work/homes/eldar.hacohen/ttexample.ttlog',
    #                       '/Neteera/Work/homes/eldar.hacohen/ttexample1.ttlog', 9500)
    # s = setupHandler._load_fmcw('/Neteera/Work/homes/eldar.hacohen/ttexample.ttlog')
    # a = 0


    # read_last_30_sec('/Neteera/Work/homes/eldar.hacohen/ttexample.ttlog')
    #TODO: create a method that does this:
    # ft_model_data.model = ft_model
    # #################################################
    # # predict on test set
    # win_sizes = [10, 30, 60, 90, 120]
    # for ws in win_sizes:
    #     test_nn(ft_model_data, X_test, y_test, ws)
    # print("TEST COMPLETE!!!")
    #
    save_path = '/Neteera/Work/homes/eldar.hacohen/python_code/Vital-Signs-Tracking/' \
                'pythondev/bioid/resources/nn_classifier'
    metadata = np.load(os.path.join(save_path, 'bioid_nn_metadata.npy'), allow_pickle=True)
    model_fn = os.path.join(save_path, 'bioid_nn_model.json')
    with open(model_fn, 'r') as f:
        ft_model_data = json.loads(f.read())
    fn = 'bioid_nn_metadata.npy'

    w = MainMenu()

    # # fn = 'bioid_nn_metadata.npy'
    # # meta_arr = [names2, ft_model_data.active_names, ft_model_data.train_setups, ft_model_data.test_setups,
    # #             setup_2_name]
    # # np.save(os.path.join(args.save_path, fn), meta_arr, allow_pickle=True)
    # # print("Metadata saved to" + os.path.join(args.save_path, fn))
    # # remove_subject_from_nn('Eldar Hacohen', metadata_dict, ft_model_data)
    #
    # ft_model_data =
    # names2 = {v for v in ft_model_data.name_2_class.items()}
    # fn = 'bioid_nn_metadata_with_ehud.npy'
    # print("Saved base NN model")
    # model_json = ft_model_data_with_ehud.model.to_json()
    # with open(ft_model_data_with_ehud.model_json_fn, "w") as json_file:
    #     json_file.write(model_json)
    # meta_arr = [names2, ft_model_data_with_ehud.active_names, ft_model_data_with_ehud.train_setups,
    #             ft_model_data_with_ehud.test_setups, setup_2_name]
    # np.save(os.path.join(args.save_path, fn), meta_arr, allow_pickle=True)
    # print("Metadata saved to" + os.path.join(args.save_path, fn))
