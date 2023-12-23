from scipy.signal import butter, lfilter
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import load_radar_data, load_radar_bins, radar_cpx_file, compute_phase
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


def zero_crossing(x: iter) -> int:
    """number of times the signal crosses the zero"""
    return ((x[:-1] * x[1:]) < 0).sum()

def estimated_range(data_matrix, rng_bin_st, dr, params) -> (float, int):
    """
    :param params: parameters
    :param data_matrix: data matrix [bins x frames]
    :param rng_bin_st: start range bin (bin offset as in tlog)
    :param dr: delta range bin
    :return: trg_rng: target range [m]
    :return: trg_rng_bin: target range idx
    """
    abs_data = np.abs(data_matrix)
    rng_var = np.var(abs_data[:, ::10], axis=1)
    rng_zc = np.array([max(200, zero_crossing(bin_data - np.mean(bin_data)))
                       for bin_data in abs_data])
    # eliminate the choice of noisy bin- (noisy= a bin with high zero crossings)
    trg_rng_bin = rng_bin_st + np.argmax(rng_var / rng_zc / rng_zc)
    return trg_rng_bin * dr, trg_rng_bin

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

import pandas as pd

home = '/Neteera/Work/homes/dana.shavit/'

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import signal
    import scipy as sp
    import beatdetection.BeatDetectionDSP as bdd
    from Tests.Utils.LoadingAPI import load_reference

    # fn = '/Neteera/Work/homes/dana.shavit/Research/210/apples_and_more_apples.csv'
    # df = pd.read_csv(fn)
    # version = df['version'].to_numpy()
    # sessions = df['session'].to_numpy()
    # configs = df['config'].to_numpy()
    # dists = df['distance'].to_numpy()
    # sns = df['sn'].to_numpy()
    fs = 500.0
    lowcut = 10
    highcut = 20

    i_medians = []
    i_max= []
    q_medians = []
    qmax = []

    mean_bin = [[],[]]
    mean_bin_zc = [[],[]]

    final_setups = []


    #sess = df['setup'].to_numpy()[i_sess]
    setups = [10389, 10369 ]#df['setup'][df['session'] == sessions[i_sess]].to_numpy()

    for b in range(15):
        fig, ax = plt.subplots(2, sharex=False)
        for i_j, j in enumerate(setups):
            final_setups.append(j)
            db.update_mysql_db(j)
            p = db.setup_dir(j)
            radar_file = radar_cpx_file(j)
            cpx, setup_fs = load_radar_data(radar_file)
            print(setup_fs)
            bins = load_radar_bins(radar_file)
            print(bins.shape, "bins.shape")


            cpx = bins[:,b]

            i = cpx.real
            #i = butter_bandpass_filter(i, lowcut, highcut, fs, order=3)
            q = cpx.imag
            #q = butter_bandpass_filter(q, lowcut, highcut, fs, order=3)

            i_mean = np.zeros_like(i)
            i_std = np.zeros_like(i)
            i_amp = np.zeros_like(i)
            i_max = np.zeros_like(i)
            len_sec = 5

            for ii in range(len_sec*500,len(i), len_sec*500):
                i_mean[ii-len_sec*500:ii] = np.mean(i[ii-len_sec*500:ii])
                i_std[ii-len_sec*500:ii] = np.std(i[ii-len_sec*500:ii])
                i_amp[ii-len_sec*500:ii] = np.max(i[ii-len_sec*500:ii]) - np.min(i[ii-len_sec*500:ii])
                i_max[ii-len_sec*500:ii] = np.max(np.abs(i[ii-len_sec*500:ii]))

            #ax[i_j].plot(i_max, label='I max', alpha=0.5, linewidth=0.5)
            ax[i_j].plot(i[int(setup_fs):int((1+len_sec)*setup_fs)], label='I', alpha=1, linewidth=0.5)
            #ax[0].plot(i_mean[:1000], label='mean 1 min' )
            #ax[0].plot(i_std[:1000], label='std 1 min' )
            #ax[i_j].plot(i_amp, label=str(j)+' '+'amp 1 min' , linewidth=0.5)

            # ax[i_j].plot(np.var(bins, axis=0)/max(np.var(bins, axis=0)), label="bin variance")
            # ax[i_j].plot(range_score/max(range_score), label="bin variance/zc^2")

            i_median = np.median(i_amp)
            q_mean = np.zeros_like(q)
            q_std = np.zeros_like(q)
            q_amp = np.zeros_like(q)
            q_max = np.zeros_like(q)
            for ii in range(len_sec*500,len(q), len_sec*500):
                q_mean[ii-len_sec*500:ii] = np.mean(q[ii-len_sec*500:ii])
                q_std[ii-len_sec*500:ii] = np.std(q[ii-len_sec*500:ii])
                q_amp[ii-len_sec*500:ii] = np.max(q[ii-len_sec*500:ii]) - np.min(q[ii-len_sec*500:ii])
                q_max[ii - len_sec * 500:ii] = np.max(np.abs(q[ii - len_sec * 500:ii]))

            #ax[i_j].plot(q_max, label='Q max', alpha=0.5, linewidth=0.5)
            #ax[i_j].plot(q[:int(len_sec*setup_fs)], label='Q', alpha=1, linewidth=0.5)
            #ax[1].plot(q_mean[:1000], label='mean 1 min')
            #ax[1].plot(q_std[:1000], label='std 1 min')
            #ax[i_j].plot(q_amp, label=str(j)+' '+'amp 1 min', linewidth=0.5)
            q_median = np.median(q_amp)
            q_medians.append(q_median)
            i_medians.append(i_median)
            ax[i_j].set_title(str(setups[i_j])+' bin '+str(b)+' first '+str(len_sec)+' seconds', fontsize=8)
            ax[i_j].grid(True)
            ax[i_j].axis('tight')
            ax[i_j].legend(loc='upper left', fontsize=6)
            mean_bin[i_j].append(np.argmax(np.var(bins, axis=0)))
            ax[i_j].tick_params(axis='both', which='major', labelsize=6)
        #plt.show()
        plt.savefig(home+"sky_ii_bin_"+str(b)+".png", dpi=1000)
        plt.close()
