import tkinter as tk
from pygame import mixer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import lfilter
import wave as wav
import struct as st


def get_biq(Bi: float, Bq: float, t: np.ndarray, noise: np.ndarray):
    pass


def w(t, Ti, Te, kb, tau, bias):
    T = Te + Ti
    peak = np.where(t >= Ti)[0][0]
    ti = t[:peak]
    te = t[peak:]
    wi = (-kb * ti**2) / (Ti*Te) + (kb * T * ti) / (Ti*Te)
    we = (kb / (1 - np.exp(-Te / tau))) * (np.exp(-(te - t[peak]) / tau) - np.exp(-Te / tau))
    return np.concatenate((wi, we)) + bias


def xhs(t, _c, _omega, _gamma):
    within_the_cos = _omega*t + _gamma*np.sin(2*t)
    return eta * np.cos(within_the_cos) * np.exp(-((t-b)**2) / _c)


def wave_to_IQ(wave):
    pass


def tk_logic():
    master = tk.Tk()
    master.title("Signal Simulator")
    master.geometry("420x200")
    master.wm_attributes('-topmost', 1)
    entries = []
    tk.Label(master, text="Heartbeats per minute").grid(row=0, column=0)
    e1 = tk.Entry(master)
    e1.grid(row=1, column=0)
    tk.Label(master, text="Respirations per minute").grid(row=0, column=2)
    e2 = tk.Entry(master)
    e2.grid(row=1, column=2)
    tk.Label(master, text="Amplitude").grid(row=2, column=0)
    e3 = tk.Spinbox(master, from_=0, to=10)
    e3.grid(row=3, column=0)
    tk.Label(master, text="Amplitude").grid(row=2, column=2)
    e4 = tk.Spinbox(master, from_=0, to=10)
    e4.grid(row=3, column=2)
    tk.Label(master, text="~Duration in seconds").grid(row=4, column=0)
    e5 = tk.Entry(master)
    e5.grid(row=5, column=0)
    tk.Label(master, text="Sampling Frequency [KHz]").grid(row=4, column=2)
    e6 = tk.Spinbox(master, from_=48, to=64)
    e6.grid(row=5, column=2)
    tk.Label(master, text="Noise Amplitude").grid(row=6, column=0)
    e7 = tk.Spinbox(master, from_=0, to=10)
    e7.grid(row=7, column=0)
    to_wav = tk.IntVar(value=0)
    e8 = tk.Entry(master)
    e8.grid(row=7, column=2)
    c1 = tk.Checkbutton(e8, text='Generate WAV', variable=to_wav, onvalue=1, offvalue=0)
    c1.pack()
    entries.append(e1)
    entries.append(e2)
    entries.append(e3)
    entries.append(e4)
    entries.append(e5)
    entries.append(e6)
    entries.append(e7)
    entries.append(e8)
    button = tk.Button(master, text='Go', command=master.quit)
    button.grid(row=8, column=1, sticky=tk.W, pady=4)
    root = tk.Tk()
    root.withdraw()
    tk.Label(master, bg='white', width=20, text='')
    tk.mainloop()
    lst = [e.get() for e in entries]
    return lst


def get_heartbeat(bpm: int, seconds: float):
    bbi = 60 / bpm
    t_hb = np.arange(signal_freq * bbi) / signal_freq
    one_hb = xhs(t_hb, c, omega, gamma)
    # all_hbs = np.tile(one_hb, num_hbs + 1)
    # return all_hbs[:len(signals)]


def normalize_signal(sig, min_val, max_val):
    min_sig = np.min(sig)
    max_sig = np.max(sig)
    sig = (sig - min_sig) / (max_sig - min_sig)
    sig *= (max_val - min_val)
    sig += min_val
    return sig.astype(int)


def make_wav_file(data):
    wav_file = wav.open('HR'+str(bpm)+'_RR'+str(rpm)+'_Amp'+str(int(2*eta))+'_'+str(Kb)+'.wav', "w")
    nchannels, sampwidth = 1, 2
    n_frames = len(data)
    comptype, compname = "NONE", "not compressed"
    wav_file.setparams((nchannels, sampwidth, int(khz) * 1000, n_frames, comptype, compname))
    for sample in data:
        wav_file.writeframes(st.pack('h', int(sample)))
    wav_file.close()


def get_end_parser(vec):
    parsers = np.where(combined_signal.astype(int) == 0)[0]
    opt1, opt2 = parsers[-1], parsers[-2]
    return opt2 if vec[opt1 - 1] <= vec[opt1] else opt1


if __name__ == '__main__':
    # Kb = 5
    Tau = 1.5
    A = 0.2
    b = 0.6
    # eta = 2
    c, omega, gamma = 0.025, 1.05, 1
    # b = phase
    # c = wavelength
    # eta = amplitude
    # omega = ?
    # gamma = aftershock drop

    # bpm, rpm, dur, khz = 60, 12, 5, 32
    [bpm, rpm, eta, Kb, dur, khz, noise_amp, if_to_wav] = tk_logic()
    eta, Kb = int(eta), int(Kb)
    bbi = 60 / int(bpm)
    inhale, exhale = (60/int(rpm)/2), (60/int(rpm)/2)
    signal_freq = int(khz) * 1000

    # generating the breath signal
    t_breath = np.arange(signal_freq * (inhale + exhale)) / signal_freq
    num_breaths = int((int(dur) / 60) * int(rpm)) + 1
    one_w = w(t_breath, inhale, exhale, Kb, Tau, 0)
    breath_signal = lfilter(np.full((4000,), 1 / 4000), 1, np.tile(one_w, 2))
    breath_signal = normalize_signal(breath_signal, -25750, 25750)
    if Kb > 0:
        parser = np.where(breath_signal.astype(int) == 0)[0][0]
    else:
        parser = np.where(breath_signal.astype(int) == -1)[0][0]
    breath_signal = np.concatenate((breath_signal[parser:], breath_signal[:parser]))[:one_w.shape[0]]
    breath_signals = np.tile(breath_signal, num_breaths)

    # generating the hb signal
    t_hb = np.arange(signal_freq * bbi) / signal_freq
    one_hb = xhs(t_hb, c, omega, gamma)
    num_hbs = int(int(dur) * int(bpm) / 60)
    hb_signal = np.tile(one_hb, num_hbs + 1)
    num = 25750 * (eta / Kb)
    hb_signal = normalize_signal(hb_signal, -num, num)
    hb_signal = hb_signal[:len(breath_signals)]
    window = np.log(np.arange(0.999, 0, -1 / ((int(dur) / 4)*signal_freq)))
    window -= np.min(window)
    window /= np.max(window)
    hb_signal = np.concatenate((hb_signal[:-len(window)], hb_signal[-len(window):] * window))
    hb_signal = hb_signal[np.where(hb_signal == 0)[0][0]:]

    # combining all together
    breath_signals = breath_signals[:len(hb_signal)]

    fig, ax = plt.subplots()
    hb_signal = lfilter(np.full((4000,), 1 / 4000), 1, hb_signal)
    # combined_signal = normalize_signal(hb_signal + breath_signals, -25750, 25750)
    combined_signal = normalize_signal(hb_signal + breath_signals, -25750, 25750)
    start_parser = np.where(combined_signal.astype(int) == 0)[0][0]
    end_parser = get_end_parser(combined_signal.astype(int))
    combined_signal = combined_signal[start_parser:end_parser]

    l, = plt.plot(np.arange(len(combined_signal)) / signal_freq, combined_signal.astype(int))
    plt.grid()
    ax.margins(x=0)
    tx = ''
    for num in combined_signal.astype(int):
        tx += str(num) + ','
    with open('outfile.txt', 'w') as f:
        f.write(tx[:-1] + '\n\n')
        f.write(str(combined_signal.shape[0]) + '\n\n')
        f.write(str(rpm)+','+str(bpm))
    if if_to_wav:
        make_wav_file(combined_signal)
    plt.show()
