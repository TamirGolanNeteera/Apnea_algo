from scipy.signal import butter, lfilter
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import load_radar_data, radar_cpx_file, compute_phase
import pandas as pd
db = DB()


spot_config = {
    'setup': {
        'maximum_window_from_end': 60,  # limit memory usage with maximum_window seconds [seconds]
        'maximum_window_from_end_bbi': 300,  # limit memory usage with maximum_window seconds
        # for beat-to-beat intervals [seconds]
        'starting_from': 0,  # skip the first `starting_from` seconds in the analysis [seconds]
    },
    'hr': {
        # --- Nonlinear filter settings ---
        'bandpass_high_cutoff': [20, 40],  # bandpass filter for heart valve signal [Hz]
        'bandpass_low_cutoff': [37 / 60, 130 / 60],  # final HR limits [Hz],
        'envelope_filter_degree': 2,
        'high_band_filter_degree': 2,

        # --- Spectrogram analysis settings ---
        'section_length': 15,  # section length used for spectrum analysis [seconds]
        'section_overlap': 13,  # overlap of sections [seconds]

        # --- Spectrogram analysis settings for optimization ---
        'section_length_optimization': 16,  # section length used for spectrum analysis [seconds]
        'section_overlap_optimization': 10,  # overlap of sections [seconds]
        'setup_quality_break_threshold': 0.85,  # break optimization loop if a decent enough setting has been found

        # --- Peak detection ---
        'threshold_clear': 0.4,  # threshold for prominence of peak detection in spectrum [-]
        'peak_prominence_weight_steepness': 8,  # weight function steepness around threshold [-]
        'threshold_quality': 0.35,  # quality score threshold parameter

        # --- Reporting ---
        # These values are based on and valid for the COVID 19 benchmark.
        'min_if_no_hr_peak': 0.96,  # fraction of spot_hr to report as lower bound if no clear hr measured
        'max_if_no_hr_peak': 1.06,  # fraction of spot_hr to report as upper bound if no clear hr measured
    },
    'rr': {
        # --- Filter settings ---
        'butter_filter_degree': 3,
        'bandpass_low_cutoff': [6 / 60, 40 / 60],  # final RR limits [Hz]

        # --- Spectrogram analysis settings ---
        'section_length': 15,  # section length used for spectrum analysis [seconds]
        'section_overlap': 13,  # overlap of sections [seconds]

        # --- Peak detection ---
        'threshold_clear': 0.65,  # threshold for prominence of peak detection in spectrum [-]
        'peak_prominence_weight_steepness': 8,  # weight function steepness around threshold [-]
        'threshold_quality': 0.40,  # quality score threshold parameter

        # --- Reporting ---
        # These values are based on and valid for the COVID 19 benchmark.
        'min_if_no_rr_peak': 0.88,  # fraction of spot_rr to report as lower bound if no clear rr measured
        'max_if_no_rr_peak': 1.12,  # fraction of spot_rr to report as upper bound if no clear rr measured
    },
    'algo_run_time_tweaks': {
        'resampling_freq_step_1': 100,  # from 500 Hz in 2 steps [Hz]
        'num_taps_fir': 51,
        'cutoff_fir': 45,
        # demodulation methods to be used
        'demodulation_methods': ['static_offset', 'complex_iq'],  # 'linear_offline'],
        # High frequency bands used in the optimization loop
        'high_frequency_bands': [[10., 30.], [20., 40.], [10., 20.]]  # , [15., 25.]]
    },
    'beat_detection': {
        'wavelet': {
            'n_taps': 128,
            'center_frequency': 15.,
            'omega': 2.8,
        },
        'highpass_wavelet': {
            'order': 4,
            'cutoff': 40 / 60,  # Hz
            'damping': 40,  # dB
            'gain_control_seconds': 1.
        },
        'minimum_reliable_percentage_fraction': 0.05,  # minimum fraction of supplied HR vector that should be reliable
        'window_length_fraction_heartbeat_interval': 0.9,  # fraction of mean heartbeat interval
        'match_Euclidean_distance': 10,  # multiple of smallest difference between any two beats
        'match_spacing_distance': 0.8,  # fraction of window length
        'full_profile_start_seconds': 5,  # determines size of full matrix profile
        'full_profile_end_seconds': 20,  # determines size of full matrix profile
        'stretch_template_before': 0.5,  # fraction of window length
        'stretch_template_after': 0.5,  # fraction of window length
        'variance_filter_length': 50,  # [samples] smoother of variance
        'variance_inlier_percentage': 0.5,  # expected fraction of inliers of the variance of heartbeats
        'variance_inlier_std_limit': 3,  # maximum deviation from mean to be considered inlier
        'center_of_mass_stretch': 10,  # samples after the center of mass to find the heartbeat 'peak'.

        'log_heartbeat_padding': 'auto',  # either 'auto' or tuple with seconds before and seconds after, eg. (0.2,0.5)
        'max_accepted_heart_rate': 160,  # Maximum accepted heart rate in Beats Per Minute.
        'min_accepted_heart_rate': 40,   # Minimum accepted heart rate in Beats Per Minute.
        'min_accepted_found_heartbeats': 0.4,  # Minimum fraction of expected number of heartbeats found.
    }
}

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_lowpass(highcut, fs, order=5):
    return butter(order,highcut, fs=fs, btype='low')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import signal
    import scipy as sp
    import beatdetection.BeatDetectionDSP as bdd
    from Tests.Utils.LoadingAPI import load_reference

    # sess = 8998
    # db.update_mysql_db(sess)
    #
    # p = db.setup_dir(sess)
    #
    # #rr_hq = np.repeat(rr_hq, fs_new)
    # radar_file = radar_cpx_file(sess)
    # cpx, setup_fs = load_radar_data(radar_file)
    #
    # phase = compute_phase(cpx)
    #
    # bins = None
    # phase_df = pd.DataFrame(phase)
    # i = cpx.real
    # q = cpx.imag
    
    # filt1535 = sp.signal.butter(6, [105, 135], 'bandpass', analog=False, fs=500, output='sos')
    # filt_06_25 = sp.signal.butter(5, [0.06, 2.5], 'bandpass', analog=False, fs=500, output='sos')
    #
    # i_flt_1535 = sp.signal.sosfilt(filt1535, i)
    # q_flt_1535 = sp.signal.sosfilt(filt1535, q)
    # phase_flt_06_25= sp.signal.sosfilt(filt_06_25, phase)
    # i_flt_1535_0625 = sp.signal.sosfilt(filt_06_25, i_flt_1535)
    # q_flt_1535_0625 = sp.signal.sosfilt(filt_06_25, q_flt_1535)
    #
    # wavelet = bdd.Morlet(
    #     spot_config['beat_detection']['wavelet']['n_taps'],
    #     spot_config['beat_detection']['wavelet']['center_frequency'],
    #     spot_config['beat_detection']['wavelet']['omega'],
    #     setup_fs,
    # )
    #
    # sos = sp.signal.cheby2(
    #     spot_config['beat_detection']['highpass_wavelet']['order'],
    #     spot_config['beat_detection']['highpass_wavelet']['damping'],
    #     spot_config['beat_detection']['highpass_wavelet']['cutoff'],
    #     btype='highpass',
    #     output='sos',
    #     fs=setup_fs,
    # )
    #
    # x_wavelet_IQ = sp.signal.sosfiltfilt(sos, np.abs(wavelet.filter(i_flt_1535_0625 + 1j * q_flt_1535_0625)))
    # filt00510 = sp.signal.butter(6, [0.05, 10], 'bandpass', analog=False, fs=500, output='sos')
    # i_flt00510 = sp.signal.sosfilt(filt00510, i)
    # q_flt00510 = sp.signal.sosfilt(filt00510, q)
    #
    # f1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # ax1.plot(x_wavelet_IQ)
    # ax2.plot(phase_flt_06_25)
    # ax3.plot(i_flt00510)
    # ax1.title.set_text("heartbeat calculation: I,Q separately,unified after bandpasses")
    # ax2.title.set_text("I after bandpass [0.66 10] Hz")
    # ax3.title.set_text("Q after bandpass [0.66 10] Hz")
    # ax1.grid()
    # ax2.grid()
    # ax3.grid()





    #plt.show()
    sess = 9825
    db.update_mysql_db(sess)

    p = db.setup_dir(sess)

    # raw_dir = os.sep.join([p, 'NES_RAW'])
    CATEGORIES = {'sleep_stages': ['W', 'N1', 'N2', 'N3', 'R'],
                  'apnea': ['normal', 'Hypopnea', 'Central', 'Obstructive', 'Mixed', 'Apnea']}

    ss_dict = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}
    p_dict = {'Prone':0, 'Supine':1, 'Left':2, 'Right':3, 'Up':-1, 'Upright':-1}
    radar_file = radar_cpx_file(sess)
    cpx, setup_fs = load_radar_data(radar_file)
    # ref = load_reference(sess, 'posture', db)
    # ref = np.array(ref.to_numpy())
    # for i,r in enumerate(ref):
    #     ref[i] = p_dict[r] if r in p_dict.keys() else -1
    # ref = np.repeat(ref, setup_fs)



    phase = compute_phase(cpx)

    bins = None
    phase_df = pd.DataFrame(phase)
    i = cpx.real
    q = cpx.imag

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 500.0
    lowcut = 110
    highcut = 120


    # Plot the frequency response for a few different orders.
    fig,ax = plt.subplots(2, sharex=True)

    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        #w, h = freqz(b, a, fs=fs, worN=2000)
        #plt.plot(w, abs(h), label="order = %d" % order)

    # Filter a noisy signal.


    # ax[0].plot(i, label='Noisy signal')
    # ax[1].plot(q, label='Noisy signal')
    #i_resp = butter_bandpass_filter(i, 0.05, 40.0/60.0, fs, order=6)
    #q_resp = butter_bandpass_filter(q, 0.05, 40.0/60.0, fs, order=6)
    i = butter_bandpass_filter(i, lowcut, highcut, fs, order=3)

    i_mean = np.zeros_like(i)
    i_std = np.zeros_like(i)
    len_sec = 30


    ax[0].plot(i, label='Filtered signal (%g Hz)' )
    i = butter_lowpass_filter(i, 1, fs, order=3)
    for ii in range(len_sec*500,len(i), len_sec*500):
        i_mean[ii-len_sec*500:ii] = np.mean(np.abs(i)[ii-len_sec*500:ii])
        i_std[ii-len_sec*500:ii] = np.std(i[ii-len_sec*500:ii])
    #ax[0].plot(i_resp, label='resp signal (%g Hz)' )
    ax[0].plot(i_mean, label='mean 1 min' )
    #ax[0].plot(i_std, label='std 1 min' )
    q = butter_bandpass_filter(q, lowcut, highcut, fs, order=3)
    q_mean = np.zeros_like(q)
    q_std = np.zeros_like(q)
    for ii in range(len_sec*500,len(q), len_sec*500):
        q_mean[ii-len_sec*500:ii] = np.mean(np.abs(q)[ii-len_sec*500:ii])
        q_std[ii-len_sec*500:ii] = np.std(q[ii-len_sec*500:ii])
    ax[1].plot(q, label='Filtered signal (%g Hz)')
    #ax[1].plot(q_resp, label='resp signal (%g Hz)')
    ax[1].plot(q_mean, label='mean 1 min')
    #ax[1].plot(q_std, label='std 1 min')
    #ax[0].xlabel('time (seconds)')
#    plt.hlines([-a, a], 0, T, linestyles='--')
    ax[0].grid(True)
    ax[0].axis('tight')
    ax[0].legend(loc='upper left')
    # ax[1].plot(ref)
    plt.show()