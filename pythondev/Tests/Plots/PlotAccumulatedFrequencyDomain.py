from Tests.Utils.PathUtils import folder_to_accumulated
from Tests.Utils.ResearchUtils import plot
from Tests.Utils.LoadingAPI import get_list_of_setups_in_folder_from_vs
from Tests.Utils.IOUtils import load

from pylibneteera.float_indexed_array import FrequencyArray

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import argparse
import imageio
import os
import numpy as np


def get_args() -> argparse.Namespace:
    """ Parse arguments"""
    parser = argparse.ArgumentParser(description='Result Evaluator Post Process')
    parser.add_argument('-folder_list', '-folders_list', metavar='folder_list', type=str, nargs='+', required=True,
                        help='list of csv files to be evaluated')
    parser.add_argument('-setups', '-session_ids', metavar='ids', nargs='+', type=int,
                        help='Setup IDs in DB to collect the online results', required=False)
    parser.add_argument('-suffix', type=str, nargs='+', default='state')
    return parser.parse_args()


def create_single_video(spectrograms, save_path):
    fig = plt.figure()
    with imageio.get_writer(save_path, mode='I') as writer:
        for sec, spec in spectrograms.items():
            canvas = FigureCanvasAgg(fig)
            plot(FrequencyArray(spec, gap=1/(2*len(spec) * 0.01)), new_fig=False, show=False,
                 label=str(sec), norm_by_max=True)
            plt.xlim([10, 300])
            canvas.draw()
            writer.append_data(np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3))
            fig.clear()
    plt.close(fig)


if __name__ == '__main__':
    args = get_args()
    folder = folder_to_accumulated(args.folder_list[0])
    
    setups = args.setups
    for suffix in args.suffix:
        for setup, path in get_list_of_setups_in_folder_from_vs(folder, suffix).items():
            if setups is None or setup in setups:
                print(f'loading {setup}')
                full_path = os.path.join(folder, path)
                states = load(full_path)
                create_single_video(states, full_path.replace('.npy', '.mp4'))
