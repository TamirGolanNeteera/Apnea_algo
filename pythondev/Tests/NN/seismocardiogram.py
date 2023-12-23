from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.getcwd()) / "src"))


from copy import deepcopy
import logging
from dataclasses import dataclass
from functools import partial, reduce
from typing import List, Optional, Callable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
import yaml
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from scipy import signal

import src.config as config
import src.data.sensors.epm as epm_file
import processors as prc
import src.dsp.profile as prf
import src.dsp.windows as win
from src.data.local import LocalData
from src.data.reference import ReferenceData
from src.db import vsms_db_api
from src.db.local import LocalSetup
from src.plot.heatmaps import plot_bin_scores, plot_quality_scores, plot_spectrogram
from src.plot import profile as ppf

config_store = ConfigStore.instance()
config_store.store(name="configuration_node", node=config.SetupConfiguration)
logger = logging.getLogger(__name__)


def detrend(x: pd.Series) -> pd.Series:
    y = x.values - x.median()
    y = signal.detrend(y)
    return pd.Series(y, index=x.index)


def median_absolute_deviation(x: pd.Series, scale=1.4826) -> float:
    return (x - x.median()).abs().median() * scale


def rolling_MAD(x: pd.Series, window: dt.timedelta) -> pd.Series:
    return x.rolling(window).aggregate(median_absolute_deviation)


def reset_index(x: pd.Series, t=None) -> pd.Series:
    if t == None:
        t = x.index[0]
    idx = (x.index - t).total_seconds()
    return pd.Series(x.values, index=idx)


MotifProcessor = Callable[[prf.Motif], prf.Motif]


def motifcompose(*functions: MotifProcessor) -> MotifProcessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def median_spectrum(motif: ppf.Motif, window=0.2) -> tuple:
    M = motif.to_frame()
    fs = prc.framerate(M)
    S = [
        signal.stft(
            x[1].values,
            fs=fs,
            nperseg=int(window * fs),
            noverlap=int((window - 0.005) * fs),
            nfft=4 * int(window * fs),
            window="hamming",
            detrend="constant",
        )
        for x in M.items()
    ]
    specs = [s[2] for s in S]
    f, t = S[0][0], S[0][1]
    return f, t, np.median(np.abs(np.dstack(specs)), axis=2)



@hydra.main(version_base=None, config_path="conf", config_name="setup_configuration")
def main(cfg: config.SetupConfiguration) -> None:

    plt.style.use(str(Path(get_original_cwd()) / "src/plot/signature.mplstyle"))

    logger.info(f"Processing {cfg.setup}")

    # Load data
    info = LocalSetup(cfg.setup)
    data = LocalData(info)

    bin_idx = np.argmax(data.bin_variance[0:])
    iq = data.data.iloc[:, bin_idx]
    iq = data.iq
    p = prc.phase_from_cpx(iq)

    DBC = vsms_db_api.DB("neteera_db")

    # Load metadata
    setup = cfg.setup
    session = DBC.session_from_setup(setup)

    # validity = DBC.setup_data_validity(setup, "nes")
    # if validity != "Confirmed":
    #     logger.error(f"Data not confirmed for setup {setup}, session {session}.")
    #     return

    sn = DBC.setup_sn(setup)
    sn = sn[0] if len(sn) == 1 else None
    version = 10 * int(sn[:2]) if sn is not None else None
    distance = DBC.setup_distance(setup)
    subject = DBC.setup_subject(setup)
    posture = DBC.setup_posture(setup)
    note = DBC.setup_note(setup)
    target = DBC.setup_target(setup)
    mount = DBC.setup_mount(setup)
    view = DBC.setup_view(setup)[0]
    gender = view["gender"]
    scenario = view["scenario"]
    ground_truth = view["gt"].split(", ")

    # Load reference data
    ref = ReferenceData(info)

    epm = ref.load("EPM_10M")
    epm.index = epm.index.tz_localize("Asia/Jerusalem").tz_convert("UTC")
    rr = epm.iloc[:, 6] / 60
    rr = pd.Series(rr.values, index=rr.index)
    rr = rr[iq.index[0] : iq.index[-1]]
    rr.name = "reference_rr"

    ecg = epm_file.EPM_BBI(info.setup)
    hb = ecg.load(info.start_time)
    hf = pd.Series([1e9 * 1 / v.astype("int") for v in hb.values], index=hb.index)

    # hr = epm.iloc[:, 0]
    # hr = pd.Series(hr.values, index=hr.index)
    # hr = hr[iq.index[0] : iq.index[-1]] / 60
    # hr.name = "reference_hr"

    # Seismocardiogram
    acceleration = prc.compose(
        prc.minus_median,
        prc.angular_velocity,
        detrend,
        partial(prc.bandpass_filter, low=8, high=20, pad=True),
        prc.differentiate,
        partial(prc.bandpass_filter, low=8, high=20, pad=True),
        partial(prc.resample, up=1, down=4),
    )(iq)

    position_lbcg = prc.compose(
        prc.minus_median,
        prc.angular_velocity,
        detrend,
        partial(prc.highpass_filter, low=1.5, pad=True),
        prc.integrate,
        partial(prc.resample, up=1, down=4),
    )(iq)

    position_bcg = prc.compose(
        prc.minus_median,
        prc.phase_from_cpx,
        detrend,
        partial(prc.bandpass_filter, low=8, high=240, pad=True),
        partial(prc.resample, up=1, down=4),
    )(iq)

    ####

    t0 = iq.index[0] + dt.timedelta(seconds=0)
    t1 = iq.index[0] + dt.timedelta(seconds=250)

    window_initial_search = dt.timedelta(milliseconds=500)
    P = prf.Profile.from_timeseries(
        acceleration[t0:t1], window_initial_search, normalize=False
    )
    Pc = deepcopy(P)
    sig = P.timeseries.abs().div(0.6745).median()
    threshold = 6 * sig
    # threshold = 0.004
    complexity1 = (
        P.complexity_annotation_vector(
            dt.timedelta(milliseconds=500), center=True, reverse=False
        )
        .clip(upper=threshold)
        .shift(-10)
        .fillna(0.0)
    )
    complexity2 = (
        P.complexity_annotation_vector(
            dt.timedelta(milliseconds=300), center=True, reverse=False
        )
        .clip(upper=threshold)
        .shift(-20)
        .fillna(0.0)
    )

    P.adjust(complexity1, scale=True)
    P.adjust(complexity2, scale=True)

    plt.close()
    fig, axs = plt.subplots(
        figsize=(15, 6),
        nrows=2,
        ncols=1,
        gridspec_kw=dict(hspace=0.1, wspace=0.25),
        squeeze=False,
        sharex=True,
    )
    axs[0, 0].plot(reset_index(P.timeseries))
    axs[0, 0].plot(reset_index(complexity1))
    axs[0, 0].plot(reset_index(complexity2))
    axs[0, 0].set_ylim(-2 * threshold, 2 * threshold)
    axs[1, 0].plot(reset_index(Pc.profile))
    axs[1, 0].plot(reset_index(P.profile))
    axs[1, 0].set_ylim(0, 0.05)
    plt.xlim(10, 30)
    # plt.xlim(16, 22)
    plt.savefig("timeseries")

    # Template search
    nmotifs = 4
    M = P.motifs(
        nmotifs,
        max_rel_distance=1.8,
        minimum_distance=dt.timedelta(milliseconds=800),
        normalize=False,
    )
    print([len(m.seqs) for m in M])


    plt.close()
    fig, axs = plt.subplots(
        figsize=(8, 2.0 * len(M)),
        nrows=len(M),
        ncols=1,
        gridspec_kw=dict(hspace=0.55, wspace=0.25),
        squeeze=True,
        sharex=True,
    )
    c_motif = next(axs[0]._get_lines.prop_cycler)["color"]
    c_median = next(axs[0]._get_lines.prop_cycler)["color"]
    c_principal = next(axs[0]._get_lines.prop_cycler)["color"]
    for i, (t, ax) in enumerate(zip(M, axs)):
        ppf.plot_motif(
            t, ax, c_motif=c_motif, c_median=c_median, plot_principal=False,
        )
        # ax.set_ylim(-3 * threshold, 3 * threshold)
        ax.set_ylim(-0.006, 0.006)
        ax.set_title(f"Motif {i+1}", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Acceleration", fontsize=8)
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.15)
    plt.savefig("motifs")

    def adjust(
        m: prf.Motif,
        x: pd.Series,
        shift: Optional[dt.timedelta] = None,
        window: Optional[dt.timedelta] = None,
    ) -> prf.Motif:
        n = deepcopy(m)
        n.adjust(x, shift=shift, window=window)
        return n

    def reduce_seqs(m: prf.Motif, rtype: str='median') -> prf.Motif:
        n = deepcopy(m)
        if rtype == 'median':
            n.seqs = [n.median()]
        if rtype == 'first':
            n.seqs = [n.seqs[0]]
        return n

    def search(
        m: prf.Motif,
        x: pd.Series,
        maxreldistance: float,
        minspacing: Optional[dt.timedelta] = None,
        normalize: bool = False,
    ) -> prf.Motif:
        template = m.seqs[0]
        matches, _ = prf.match_template(
            x=x, 
            template=template, 
            maxreldistance=maxreldistance, 
            minspacing=minspacing, 
            normalize=normalize)
        return prf.Motif([x[idx : idx + template.index[-1]] for idx in matches])

    def plot_motif(m: prf.Motif, savename: str, title: Optional[str]=None, normalize=False, axs=None, ylim=None) -> prf.Motif:
        plt.close()
        if axs is None:
            fig, axs = plt.subplots(
                figsize=(8, 6),
                nrows=1,
                ncols=1,
                gridspec_kw=dict(hspace=0.1),
                squeeze=False,
                sharex=True,
            )
        c_motif = next(axs[0,0]._get_lines.prop_cycler)["color"]
        c_median = next(axs[0,0]._get_lines.prop_cycler)["color"]
        ppf.plot_motif(m, axs[0,0], plot_median=True, plot_principal=False, c_median=c_median, c_motif=c_motif, normalize=normalize)
        plt.xlabel("Time (s)")
        if normalize:
            plt.ylim(-2.5, 2.5)
        else:
            df = m.to_frame()
            sigma = df.abs().median(axis=0).quantile(0.8) / 0.6745
            if ylim:
                plt.ylim(-ylim, ylim)
            else:
                plt.ylim(-8*sigma, 8*sigma)
        if title:
            plt.title(title)
        else:
            plt.title(f"Template and template matches (n={len(m.seqs)})")
        plt.legend()
        plt.subplots_adjust(bottom=0.10, top=0.90, left=0.15)
        plt.savefig(savename)
        return m

    def print_len(m):
        print(len(m.seqs))
        return m

    def plot_spectrogram(m: prf.Motif, savename: str, title: Optional[str]=None) -> prf.Motif:
        f, t, S = median_spectrum(m)
        plt.close()
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10, 9),
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": [1], "hspace": 0.25},
        )

        ax[0, 0].pcolormesh(t, f, np.log1p(np.sqrt(S)), shading="nearest")
        ax[0, 0].set_ylabel("Frequency (Hz)")
        ax[0, 0].set_xlabel("Time (s)")
        if title:
            ax[0, 0].set_title(title)
        else:
            ax[0, 0].set_title("Median spectrogram")
        plt.subplots_adjust(bottom=0.10, top=0.90, left=0.15)
        plt.savefig(savename)
        return m

    for i, m in enumerate([m for m in M if len(m.seqs) > 1]):
        c = m.center_of_mass()
        w = dt.timedelta(milliseconds=780)

        # Find motif and matches
        m = motifcompose(
            partial(
                adjust, 
                x=P.timeseries, 
                shift=dt.timedelta(milliseconds=-200) + c, 
                window=w),
            partial(plot_motif, savename=f"motif_{i}_setup_{cfg.setup}", ylim=0.01),
            partial(reduce_seqs, rtype='median'),
            partial(search, x=P.timeseries, maxreldistance=1.8, normalize=False, minspacing=dt.timedelta(milliseconds=600)),
            partial(plot_motif, savename=f"motif_{i}_matches_setup_{cfg.setup}", ylim=0.005),
        )(deepcopy(m));

        n = motifcompose(
            partial(
                    adjust, 
                    x=position_bcg, 
                    shift=dt.timedelta(milliseconds=-300), 
                    window=w + dt.timedelta(milliseconds=600)),
            partial(plot_motif, savename=f"motif_{i}_acceleration_setup_{cfg.setup}", ylim=0.4),
        )(m);
    
        # Cut out position
        p = motifcompose(
            partial(
                    adjust, 
                    x=position_bcg, 
                    shift=dt.timedelta(milliseconds=-300), 
                    window=w + dt.timedelta(milliseconds=600)),
            partial(plot_motif, savename=f"motif_{i}_position_setup_{cfg.setup}", ylim=0.4),
        )(m);

        # Cut out bcg
        b = motifcompose(
            partial(
                    adjust, 
                    x=position_lbcg, 
                    shift=dt.timedelta(milliseconds=-300), 
                    window=w + dt.timedelta(milliseconds=600)),
            partial(plot_motif, savename=f"motif_{i}_bcg_setup_{cfg.setup}", ylim=4),
        )(m);

        plot_spectrogram(
            n, 
            savename=f"motif_{i}_median_spectrogram_setup_{cfg.setup}",
            title=f"Median spectrogram of seismocardiogram, setup {cfg.setup}",
            )
        
        n.to_frame().to_csv(f"setup_{cfg.setup}_motif_{i}_acceleration.csv")
        p.to_frame().to_csv(f"setup_{cfg.setup}_motif_{i}_position.csv")
        b.to_frame().to_csv(f"setup_{cfg.setup}_motif_{i}_bcg.csv")

    return


if __name__ == "__main__":
    main()
