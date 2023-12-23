# Copyright (c) 2021 Neteera Technologies Ltd. - Confidential
import os
import sys
sys.path.insert(1, os.getcwd())

import argparse
import numpy as np
from Tests import vsms_db_api as db_api


def get_args() -> argparse.Namespace:
    """ Argument parser

    :return: parsed arguments of the types listed within the function
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-session_ids', metavar='ids', nargs='+', type=int, help='Setup IDs in DB', required=False)
    parser.add_argument('-fs', metavar='seed', type=int, required=False, help='New sampling rate')
    parser.add_argument('-nsec', metavar='seed', type=int, required=False, help='Signal length in seconds')
    parser.add_argument('-save_path', metavar='SavePath', type=str, help='Path to save data', required=True)
    parser.add_argument('-target', metavar='Target', type=str, help='back or front', required=True)

    return parser.parse_args()


def circlecenter(x, y):
    """Given points on a circle with coordinates (x,y), compute the center of the circle."""
    f = 10  # controls numerical accuracy. Some number same-ish order of magnitude as I and Q.
    Nx = len(x)

    # Correlation of data
    Mxy = np.vstack(
        (
            np.square(x) + np.square(y),
            2 * f * x,
            2 * f * y,
            f**2 * np.ones(Nx),
        )
    )
    mn = np.mean(Mxy, axis=1)
    Cmx = np.cov(Mxy)
    X = Cmx + np.outer(mn, mn)

    xc = np.mean(x)
    yc = np.mean(y)
    sxs = np.var(x)
    sys = np.var(y)
    sx = sxs + xc**2
    sy = sys + yc**2
    N = np.vstack(
        (
            [4 * (sx + sy), (2 * f * xc), (2 * f * yc), 0],
            [(2 * f * xc), (f**2), 0, 0],
            [(2 * f * yc), 0, (f**2), 0],
            [0, 0, 0, 0],
        )
    )
    D, V = linalg.eig(X, N)
    i = np.argsort(np.abs(D))
    v = V[:, i[0]]
    if v[0] != 0.0:
        v = v / v[0]

    xch = -v[1] * f / v[0]
    ych = -v[2] * f / v[0]

    return xch, ych


def radar_cpx_file(setup: int) -> str:
    tlog_files_radar = db.setup_ref_path(setup=setup, sensor=db_api.Sensor.nes)
    dir_radar = os.path.dirname(tlog_files_radar[0])
    return glob.glob(os.path.join(dir_radar, "*.npy"))[0]


def load_radar_data(filename: str) -> Tuple[np.ndarray, float]:
    X, Ts = np.load(filename, allow_pickle=True)[[0, 2]]
    x = X[:, np.argmax(np.var(X, axis=0))]
    return x, 1 / Ts\

def compute_phase(iq: np.ndarray) -> np.ndarray:
    """Return the phase of the radar signal after removing the bias."""
    cr, ci = circlecenter(np.real(iq), np.imag(iq))
    return np.unwrap(np.angle(iq - (cr + 1j * ci)))


if __name__ == '__main__':

    args = get_args()
    db = db_api.DB()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.ssave_path)
    col = ['gray', 'blue', 'green', 'red', 'yellow', 'magenta', 'cyan']

    setups = db.all_setups()
    setups = [s for s in setups if s > 10000]
    setups = [s for s in setups if db.setup_distance(s)==1000]
    setups = [s for s in setups if db.setup_sn(s) and db.setup_sn(s)[0][0]=='2']


    """Iterate over all sessions to create data"""


    for i_sess, sess in enumerate(setups):
        db.update_mysql_db(sess)

        print(":::::::: processing session", sess, str(i_sess) + '/' + str(len(setups)), "::::::::")
        radar_file = radar_cpx_file(sess)
        iq_data, setup_fs = load_radar_data(radar_file)
        phase = compute_phase(iq_data)



        print("successfully created AHI labels")
        np.save(os.path.join(save_path,str(sess) + '_y.npy'), y, allow_pickle=True)
        #np.save(os.path.join(save_path,str(sess) + '_y3.npy'), y3, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_X.npy'), X, allow_pickle=True)
        np.save(os.path.join(save_path,str(sess) + '_valid.npy'), valid, allow_pickle=True)

        print("saved training data")
