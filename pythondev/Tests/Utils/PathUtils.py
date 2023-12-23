#   utils for file paths and more based on os module

from Tests.Constants import DELIVERED
from Tests.vsms_db_api import Benchmark, Project

import os
import platform
import re
from distutils.dir_util import copy_tree
import numpy as np


def create_dir(directory: str) -> bool:
    """ Create a directory with a given path if none exists
    :param str directory: directory path
    """
    if directory:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            return True
    return False


def windows_dir_to_linux_dir(path: str) -> str:
    if r'N:' in path and platform.system() == 'Linux':
        path = path.replace(r'N:', '/Neteera/Work').replace("\\", "/")
    return path


def change_dir_to_pythondev():
    cwd = os.getcwd()
    if 'pythondev' in cwd:
        pythondev_folder = 'pythondev'
    else:
        try:
            pythondev_folder = re.search(r'net-alg-[\d.]+', cwd)[0]
        except TypeError:
            splitted = np.array(cwd.split(os.sep))
            pythondev_folder = os.sep.join(splitted[:np.argwhere(splitted[1:] == splitted[:-1])[0][0] + 2])
    pythondev_ends = cwd.rfind(pythondev_folder) + len(pythondev_folder) + 1
    if pythondev_ends < len(cwd):
        os.chdir(cwd[:pythondev_ends])
        print(f'working dir changed to {os.getcwd()}')


def path_dirname_multiple(path: str, num_folders_to_go_upper: int) -> str:
    out = path
    for _ in range(num_folders_to_go_upper):
        out = os.path.dirname(out)
    return out


def copy_code_and_change_working_dir(folder):
    copy_tree(path_dirname_multiple(__file__, 3), os.path.join(folder, 'pythondev'))
    os.chdir(os.path.join(folder, 'pythondev'))
    print(f'working dir changed to {os.getcwd()}')


def generate_plot_path(folder, vs, idx):
    file_name = f'{vs}_{idx}.png'
    out = f'=HYPERLINK(\"{os.path.join(folder, file_name)}\", \"{vs}_{idx}\")'.replace(
        '/Neteera/Work', r'N:').replace("/", "\\")
    if vs in ['hr', 'rr']:
        out = out.replace('spot', 'dynamic')
    return out


def remove_slash_from_file_path(path):
    if path.endswith(os.sep):
        return path[:-1]
    else:
        return path


def add_mode_to_path(path, mode):
    if path.endswith(mode):
        return path
    else:
        return os.path.join(path, mode)


def get_all_folders_same_mode(folders):
    for folder in folders:
        inner_dir = os.path.basename(folder)
        if inner_dir in ['dynamic', 'spot']:
            return [add_mode_to_path(fol, inner_dir) for fol in folders]
    return folders


def get_folder_list(input_folder_list):
    """ if the input (single) folder contains multiple versions so compare between the inner versions(folders).
    Also, convert windows path to linux path if running with linux"""
    out_path = None
    out_list = [remove_slash_from_file_path(windows_dir_to_linux_dir(folder)) for folder in input_folder_list]
    out_list = get_all_folders_same_mode(out_list)

    input_folder = out_list[0]
    list_dir = os.listdir(input_folder)
    if len(out_list) == 1 and 'dynamic' not in list_dir and 'spot' not in list_dir and 'dynamic' not in input_folder \
            and 'spot' not in input_folder and len(list_dir):
        out_list = []
        list_dir = list_dir
        if 'online' in list_dir:
            list_dir.pop(list_dir.index('online'))
            list_dir = ['online'] + list_dir
        for file in list_dir:
            full_path = os.path.join(input_folder, file)
            if os.path.isdir(full_path):
                list_dir_inner = os.listdir(full_path)
                if 'spot' in list_dir_inner or 'dynamic' in list_dir_inner:
                    out_list.append(full_path)
        out_path = input_folder
    released = os.listdir(DELIVERED)
    bench = get_benchmark_from_path(input_folder)
    for i, folder in enumerate(out_list):
        if folder in released:
            out_list[i] = os.path.join(DELIVERED, folder, 'stats', bench)
            if os.path.basename(input_folder) in ['dynamic', 'spot']:
                out_list[i] += os.path.basename(input_folder)
    return out_list, out_path


def find_benchmarks_from_folders(folder_list):
    benchmark_str = ''
    for folder in folder_list:
        bench = get_benchmark_from_path(folder)
        if bench is not None and bench not in benchmark_str:
            benchmark_str += f'_{bench}'
    return benchmark_str


def get_benchmark_from_path(path):
    for bench in {str(x) for x in Benchmark} | {str(x) for x in Project}:
        if bench in path:
            return bench


def remove_sep_at_end(path):
    if path.endswith(os.sep):
        return path[:-1]
    else:
        return path


def folder_to_accumulated(folder_inp):
    folder_out = remove_sep_at_end(folder_inp)
    if not folder_out.endswith('accumulated'):
        if folder_out.endswith('spot') or folder_out.endswith('dynamic'):
            folder_out = folder_out.replace('spot', 'accumulated').replace('dynamic', 'accumulated')
        else:
            folder_out = os.path.join(folder_out, 'accumulated')
    return folder_out
