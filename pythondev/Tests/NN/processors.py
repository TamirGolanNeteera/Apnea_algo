import datetime as dt
from functools import reduce
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.signal as signal
from dsp.circlecenter import circlecenter

Processor = Callable[[pd.Series], pd.Series]
Reducer = Callable[[pd.Series], Any]
TimeReduction = Callable[[pd.Series], pd.Timestamp]


def timed_reducer(time_reduction: TimeReduction, reducer: Reducer) -> Processor:
    """Combine a function that computes a value from a time series, with a function that picks the time to which the value is assigned"""
    func: Processor = lambda x: pd.Series(
        [reducer(x)], [time_reduction(x.index.to_series())]
    )
    return func


def FirstTime(x: pd.Series) -> pd.Timestamp:
    return x.iloc[0]


def LastTime(x: pd.Series) -> pd.Timestamp:
    return x.iloc[-1]


def MedianTime(x: pd.Series) -> pd.Timestamp:
    return x.median()


def compose(*functions: Processor) -> Processor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def framerate(x: pd.Series) -> float:
    return 1_000_000_000 / int(x.index.to_series().diff().median().to_numpy())


def phase_from_cpx(x: pd.Series) -> pd.Series:
    c = circlecenter(x.to_numpy())
    return pd.Series(np.unwrap(np.angle(x.to_numpy() - c)), x.index)


def bandpass_filter(x: pd.Series, low, high, order=3, pad=False) -> pd.Series:
    sos = signal.butter(
        order,
        [low, high],
        btype="bp",
        output="sos",
        fs=framerate(x),
    )
    if not pad:
        y = signal.sosfiltfilt(sos, x.to_numpy())
        return pd.Series(y, x.index)
    else:
        pad = x.values[::-1]
        z = np.hstack((pad, x.values, pad))
        y = signal.sosfiltfilt(sos, z)
        y = y[len(pad) : 2 * len(pad)]
        return pd.Series(y, x.index)


def highpass_filter(x: pd.Series, low: float, order=3, pad=False) -> pd.Series:
    sos = signal.butter(
        order,
        low,
        btype="highpass",
        output="sos",
        fs=framerate(x),
    )
    if not pad:
        y = signal.sosfiltfilt(sos, x.to_numpy())
        return pd.Series(y, x.index)
    else:
        pad = x.values[::-1]
        z = np.hstack((pad, x.values, pad))
        y = signal.sosfiltfilt(sos, z)
        y = y[len(pad) : 2 * len(pad)]
        return pd.Series(y, x.index)


def lowpass_filter(x: pd.Series, high: float, order=3, pad=False) -> pd.Series:
    sos = signal.butter(
        order,
        high,
        btype="lowpass",
        output="sos",
        fs=framerate(x),
    )
    if not pad:
        y = signal.sosfiltfilt(sos, x.to_numpy())
        return pd.Series(y, x.index)
    else:
        pad = x.values[::-1]
        z = np.hstack((pad, x.values, pad))
        y = signal.sosfiltfilt(sos, z)
        y = y[len(pad) : 2 * len(pad)]
        return pd.Series(y, x.index)


def rectify(x: pd.Series) -> pd.Series:
    return x.abs()


def welch(x: pd.Series, **kw) -> pd.Series:
    f, P = signal.welch(x.to_numpy(), **kw)
    return pd.Series(P, index=f)


def resample(x: pd.Series, up: int, down: int, **kw) -> pd.Series:
    def timeindex(start: dt.datetime, fs: float, periods=int) -> pd.DatetimeIndex:
        Ts = dt.timedelta(seconds=1 / fs)
        return pd.date_range(start=start, periods=periods, freq=Ts)

    fs = framerate(x)
    x_resampled = signal.resample_poly(x.to_numpy(), up, down, **kw)
    t_resampled = timeindex(x.index[0], fs * up / down, len(x_resampled))

    return pd.Series(x_resampled, t_resampled)


def standard_deviation(x: pd.Series) -> float:
    return x.std()


def maximum(x: pd.Series) -> float:
    return x.max()


def differentiate(x: pd.Series) -> pd.Series:
    return x.diff().fillna(0.0) * framerate(x)


def zero_crossing(x: pd.Series, detrend=True) -> int:
    y = x if detrend else pd.Series(x.values - signal.detrend(x.values), index=x.index)
    y = y.diff()
    return pd.Series(y.abs().values > 0, index=y.index)


def tanh(x: pd.Series) -> pd.Series:
    return pd.Series(np.tanh(x.values), index=x.index)


def autoscale(x: pd.Series, q=0.25) -> pd.Series:
    scale = (
        x.quantile(1 - q) - x.quantile(q)
    ) / 1.349  # see wiki Robust Measures of Scale
    y = pd.Series((x.values - x.quantile(0.5)) / (2 * scale), index=x.index)

    return tanh(y)


def integrate(x: pd.Series) -> pd.Series:
    return x.cumsum() * framerate(x)


def prominence(x: pd.Series, normalize=True) -> float:
    y = x / x.sum() if normalize else x
    ipks, prom = signal.find_peaks(y.values, prominence=0.0001, height=0.0)
    if len(ipks) == 0:
        return 0.0
    index_max_peak = np.argsort(prom["peak_heights"])[-1]
    return prom["prominences"][index_max_peak]


def prominence_quality(
    prom: float, threshold: float = 0.5, steepness: float = 8.0
) -> float:
    return (np.tanh(steepness * (prom - threshold)) + 1) / 2


def real_only(x: pd.Series) -> pd.Series:
    return pd.Series(np.real(x.values), index=x.index)


def imag_only(x: pd.Series) -> pd.Series:
    return pd.Series(np.imag(x.values), index=x.index)


def angular_velocity(x: pd.Series) -> pd.Series:
    i, q = real_only(x), imag_only(x)
    fs = framerate(x)
    di, dq = i.diff().fillna(0.0) / fs, q.diff().fillna(0.0) / fs
    return pd.Series((i * dq - q * di) / (i * i + q * q + 1), index=x.index)


def minus_median(x: pd.Series) -> pd.Series:
    if type(x[0]) == np.float64:
        mi = np.median(x.values)
    else:
        mi = np.median(np.real(x.values)) + 1j * np.median(np.imag(x.values))
    return x - mi


def detrend(x: pd.Series) -> pd.Series:
    y = x.values - x.median()
    y = signal.detrend(y)
    return pd.Series(y, index=x.index)
