
import numpy as np
import scipy.signal as sp
from Tests.Utils.DBUtils import *

db = DB()


def moving_average(x, N, compensate_phase_delay=True):
    #print(x.shape)
    n = x.shape[0]
    y = np.hstack((x[::-1], x, x[::-1]))
    # Compensate for phase delay of a moving average filter
    if compensate_phase_delay:
        z = sp.lfilter(np.ones(N) / N, np.asarray([1]), y)
        zc = z[n:2 * n]
        #print(n, zc.shape)
        m = int(np.round((N+1)/2))
        v = np.hstack((zc[m:], zc[-1]*np.ones(m)))
        assert n == v.shape[0]
    else:
        v = sp.filtfilt(np.ones(N) / N, np.asarray([1]), y)[n:2 * n]
    return v


def features(x, fs, normalize=True):
    # new_x = np.hstack((x[::-1], x, x[::-1]))
    fn = 500.
    fc = [5., 6., 7., 8., 9., 11., 13., 15., 6.]
    Om = [4., 4., 4., 5., 5., 5., 6., 6., 6.]
    z = []
    for fci, Omi in zip(fc, Om):
        width = Omi * fs / (2 * np.pi * fci)
        y = sp.cwt(x, sp.morlet2, [width], w=Omi)
        z.append(np.abs(y))
    up = int(20*fn/fs)
    down = 20
    zr = [np.asarray(sp.resample_poly(zi, up, down)) for zi in z]
    ma = [moving_average(np.squeeze(x), int(fn*2.5)) for x in zr]
    if normalize:
        r = [np.squeeze(x1 / x2) for (x1,x2) in zip(zr, ma)]
    else:
        r = [np.squeeze(x1) for (x1, x2) in zip(zr, ma)]
    b, a = sp.butter(3, [40/50, 300/60], btype='bandpass', fs=fn)
    y = sp.filtfilt(b, a, r[:-1], padlen=1500)

    tau1 = int(fn * 0.1)
    tau2 = int(fn * 0.6)
    # y = y[:, int(y.shape[1]/3):-int(y.shape[1]/3)]
    hb_idx = np.where(np.diff(np.sign(y[0])) > 0)
    hb_idx = hb_idx[0][np.nonzero(hb_idx[0] - tau1 > 0)]
    hb_idx = hb_idx[np.nonzero(hb_idx + tau2 < len(y[0]))]
    return y, hb_idx

def front_features(x, fs, normalize=True):
    # new_x = np.hstack((x[::-1], x, x[::-1]))
    fn = 500.
    fc = [5., 6., 7., 8., 9., 11., 13., 15., 6.]
    Om = [4., 4., 4., 5., 5., 5., 6., 6., 6.]
    z = []

    for fci, Omi in zip(fc, Om):
        width = Omi * fs / (2 * np.pi * fci)
        y = sp.cwt(x, sp.morlet2, [width], w=Omi)
        z.append(np.abs(y))
    up = int(20*fn/fs)
    down = 20
    #zr = [np.asarray(sp.resample_poly(zi, up, down)) for zi in z]

    zr = [np.asarray(sp.resample_poly(zi, up, down)) for zi in z]
    ma = [moving_average(np.squeeze(x), int(fn * 2.5)) for x in zr]
    if normalize:
        r = [np.squeeze(x1 / x2) for (x1, x2) in zip(zr, ma)]
    else:
        r = [np.squeeze(x1) for (x1, x2) in zip(zr, ma)]

    b, a = sp.butter(3, [40/50, 300/60], btype='bandpass', fs=fn)
    y = sp.filtfilt(b, a, r[:-1], padlen=1500)

    return y, np.arange(0, y.shape[1], fs) #signal every second


def spectrogram_features(x, fs):
    import matplotlib.pyplot as plt
    #min_i = np.min(r.data['i'])
    #min_q = np.min(r.data['q'])
    #c = (min_i - 1E4) + 1j * (min_q - 1E4)
    #phi = np.angle(r.data['i'] + 1j * r.data['q'] - c)
    # sos = sp.cheby2(
    #     4,
    #     40,
    #     0.5,
    #     btype='highpass',
    #     output='sos',
    #     fs=fs,
    # )
    # y = sp.sosfiltfilt(sos, x)
    # Compute frequency vector, time vector and magnitude of the Fourier transforms
    overlap = 30
    nsec = 60
    f, t, Sxx = sp.spectrogram(x, fs=fs, nperseg=60*fs, noverlap=30*fs, mode='magnitude')
    # Limit the frequencies (we don't need all 250Hz)
    fimax = np.where(f > 30)[0][0] ## Max 30 Hz
    f = f[0:fimax]
    Sxx = Sxx[0:fimax, :]
    plt.pcolormesh(np.log10(Sxx))

    return Sxx, np.arange(0, Sxx.shape[0], 1)