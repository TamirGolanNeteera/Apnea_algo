# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

from pylibneteera.float_indexed_array import FrequencyArray

import numpy as np
import copy
from itertools import cycle


back_chair_config = {'hr': {'win_sec': 10,  # base configurations, other mounts are copied from this one
                            'compute_frames': cycle([True]),
                            'band': (35 / 60, 200 / 60),
                            'kalman':
                                {'initial_parameters': {
                                    'band': (35 / 60, 200 / 60),
                                    'max_change': 4,   # bpm
                                    'max_change_low': 1,  # bpm
                                    'gain_by_addition': 0.006,  # 0 <= gain <= 1
                                    'gain_by_multiply': 0.03,  # 0 <= gain <= 1
                                    'close_freqs_margin': 0.1666,  # 10 bpm
                                    'initial_prediction': 70,  # bpm
                                    'observation_history_long': 10,    # sec
                                    'observation_history_short': 8,    # sec
                                    'high_freq_to_search_for_halves': 1.583,   # 95 bpm
                                    'peak_height_of_half_harmonic': 0.7,
                                    'second_peak_height_to_dismiss_state': 0.8,   # relative
                                    'max_min_margin_to_boost_gain': 10,    # bpm
                                    'gain_boost_factor': 3,
                                    'random_generator': [0,  1, 0,  1,  0, -1, -1, 0, -1,  0,  1, 0, -1, -1, 0, -1, -1,
                                                         0, -1,  1,  0,  1,  0,  0,  0,  1, -1,  0, -1, -1],
                                    # hard coded to be with a gap of "to_freq / spectrogram_fft_size
                                    # the length should be spectrogram_fft_size / 2"
                                    'initial_state':
                                        FrequencyArray([0]*52 + [0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031,
                                                                 0.034, 0.037, 0.042, 0.047, 0.053, 0.059, 0.065, 0.071,
                                                                 0.077, 0.082, 0.088, 0.094, 0.1, 0.106, 0.111, 0.117,
                                                                 0.123, 0.129, 0.134, 0.14,
                                                                 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16,
                                                                 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
                                                                 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.15, 0.15,
                                                                 0.15, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14,
                                                                 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,
                                                                 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                                                                 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.1,
                                                                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09,
                                                                 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08,
                                                                 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07,
                                                                 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
                                                                 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04,
                                                                 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                                                 0.04, 0.04, 0.03] + 3857 * [0], gap=100/8192)}},
                            'folding': {
                                'phase_harmonic_weights': {1: 1, 2: 1, 3: 1},
                                'amplitude_harmonic_weights': {1: 1, 2: 0.3, 3: 0.3},
                                'min_hr_to_fold_amplitude_ft': 90,    # bpm
                            },
                            'reliability_params': {'min_time_for_reliability': 10,     # sec
                                                   'min_duration': 8,  # sec
                                                   'min_readings': 4,
                                                   'max_pred_variance': 9,  # bpm
                                                   'max_noise': 4,  # sec
                                                   'no_reliable_time_to_stop_output': 90,
                                                   'no_reliable_time_to_stop_high_quality': 15,
                                                   'harsh_motion_time_to_stop_output': 10,
                                                   'pred_measure_max_diff': 8},
                            'prep': {'to_freq': 100,   # Hz
                                     'high': 15,   # Hz
                                     'low': 35 / 60,  # Hz
                                     'loc_low_for_low_observation': 50 / 60,  # Hz
                                     'low_observation': 60,  # bpm
                                     'filter_order': 4,
                                     'num_taps_fir': 51,
                                     'cutoff_fir': 45,  # bpm
                                     'initial_sampling_rate_cut': 100,  # bpm
                                     'high_filter_pass_method': 'lfilter',
                                     'high_filter_pass_method_2nd': 'filtfilt',
                                     'lfilter_reduction_tail': 0.15,   # 15%
                                     # for standing this is 3 although it is filtfilt (optimized)
                                     'high_filter_degree': 3,
                                     'high_filter_degree_2nd': 1,

                                     },
                            'rest_motion_toggle_parameters': {
                                'to_freq_decimation_for_phase_derivative': 5,  # Hz
                                'low_hr': 65,  # bpm
                                'max_median_rolling': 200,  # relative
                                'max_min_percentile': 65,
                                'high_iq_change': 20e3,    # iq values
                                'very_low_hr': 50,  # bpm
                                'number_of_peaks_to_dismiss_low_hr_readings': 3,
                                'peak_heights_to_dismiss_low_hr_readings': 0.7,  # relative
                                'maximal_derivative': 12,  # rad/sec
                                'maximal_derivative_high': 16,  # rad/sec
                                'high_prominence': 0.8,  # relative
                                'low_prominence': 0.55,  # relative
                                'rr_threshold_autocorr': 20,   # bpm
                            },
                            'validity': {
                                'lower_limit': 40,  # bpm
                                'deviation_from_rr_or_2nd_harmonic': 7,  # bpm
                                'deviation_from_2rd_harmonic': 5,   # bpm
                                'ra_for_dismiss_3rd_harmonic': 20},  # rad
                            'regular_phase_preprocess_quality_advantage': 0.1,
                            'spectrogram_fft_size': 8192,
                            'default_band': [10, 20],  # Hz
                            },
                     'rr': {'win_sec': 20,
                            'min_win_sec': 10,
                            'compute_frames': cycle([True, False]),
                            'kalman':
                                {'initial_parameters': {'initial_state': -1,
                                                        'initial_error': 15,
                                                        'noise_propagation': 0.01,
                                                        'max_change': 2  # bpm
                                                        },
                                 'power': -4,
                                 'factor': 0.2
                                 },
                            'reliability_params': {'min_time_for_reliability': 10,
                                                   'max_noise': 6,
                                                   'min_readings': 5,
                                                   'min_duration': 10,
                                                   'no_reliable_time_to_stop_output': 90,  # sec
                                                   'no_reliable_time_to_stop_high_quality': 15,  # sec
                                                   'harsh_motion_time_to_stop_output': 10,   # sec
                                                   'time_to_output_without_high_quality': 30,  # sec
                                                   'max_pred_variance': 2,  # bpm
                                                   },
                            'prep': {'to_freq': 10,    # Hz
                                     'num_taps_fir': 51,
                                     'cutoff_fir': 3,  # Hz
                                     'high': 45. / 60,  # Hz
                                     'low': 0.05,  # Hz
                                     'filter_order': 3},
                            'rr_autocorr_parameters': {
                                'prediction_upper_limit': 40,  # bpm
                                'prediction_lower_limit': 4,  # bpm
                                'prediction_upper_limit_for_amplitude': 23,  # bpm
                                'quality_loss_for_high_rr_phase': 0.15,
                                'phase_quality_advantage': 0.2,
                                'autocorr_noise_thresh': 0.3,
                                'relative_height': 0.25,
                                'min_rr_times_window': 100,   # bpm * s
                                'max_2nd_derivative_zero_crossings': 6}

                            },
                     'stat': {'win_sec': 10,
                              'ec_window': 3,  # sec
                              'compute_frames': cycle([True]),
                              'largest_breath': 2,  # radians
                              'acc_ec_win_size': 4,
                              'acc_ec_thresh': 3,
                              'interval_len_in_sec': 0.5,  # sec
                              # 70 crossings in 500 Hz * 0.5 sec samples
                              'number_of_zero_crossing_for_empty': 70 / 500 / 0.5,
                              'number_zero_crossing_low': 35 / 500 / 0.5,
                              'border_for_very_high_phase_when_getting_up': 7,     # sec
                              'first_section_low_phase_difference': 100,   # rad
                              '2nd_section_high_phase_difference': 80,  # rad
                              'ratio_std_start_end_iq': 2,
                              'ft_max': 2000,
                              'zero_crossings_for_low_ft_max': 50,
                              'percentile_zero_crossing': 30,     # %
                              # todo normalize by radar configurations (sample number, radar gain, etc)
                              'iq_max_diff': 5000,
                              # todo normalize by radar configurations (sample number, radar gain, etc)
                              'max_min_i_q': 500,
                              'percentile_phase_derivative': 90,   # %
                              'rr_qual_thresh': 0.7,
                              'model_path': 'models/rr_svm_status_back.model',
                              'model_parameters': {
                                  'coefficients': np.array([-0.5777960, -2.993963]),
                                  'intercept': -2.4466539,
                                  'scale': np.array([0.22769, 0.10317]),
                                  'mean': np.array([0.58174, 0.06991])},
                              'prep': {'to_freq': 10,
                                       'num_taps_fir': 51,
                                       'cutoff_fir': 3,
                                       'high': 45. / 60,
                                       'low': 0,
                                       'filter_order': 3},
                              'motion': {
                                  'phase_derivative': 16,
                                  'to_freq_decimation_for_phase_derivative': 5,  # Hz
                                  'win_sec': 1,    # sec
                              }
                              },
                     'ra': {'compute_frames': cycle([True] + 3 * [False])},
                     'ie': {
                         'win_sec': 40,
                         'min_win_sec': 32,
                         'compute_frames': cycle([True] + 3 * [False]),
                         'lowpass_filter_degree': 4,
                         'lowpass_filter_cutoff': 42 / 60,  # Hz
                         'threshold_factor': 0.3,  # influences dynamic threshold to separate in- and exhale.
                         'average_over_ratios': 4,
                         'resample_frequency': 10,  # Hz
                     },
                     'intra_breath': {'win_sec': 1,
                                      'compute_frames': cycle([True]),
                                      'compute_every_part_of_sec': 0.1,  # compute every 0.1 sec (100 ms)
                                      'filter_cutoff': 0.5,  # todo should be changed to work with rr > 30 bpm
                                      'filter_degree': 2,  # changed for other mounts
                                      'phase_derivative_threshold': 0.01},  # changed for other mounts
                     'range_config': {
                         'estimate_range': True,
                         'win_sec': 6,
                         'stride': 10,
                         'bottom_clip_zero_crossing': 200,     # calibrated to 500 Hz signal
                         'constant_bin': 2,
                         # degenerated, can be set to 1 for a specific trial
                         'maximum_distance_in_bins_from_default': 10000,
                         'delta_to_re_calc': 20,
                         'min_bin': 2,
                         'min_bin_when_empty': 3,
                         'max_distance': 1800,    # mm # changed for front
                         'min_bin_diff': 2,
                         'time_to_detect': 6},
                     'fmcw_raw_params': {'TC': 68,
                                         'nfft': 256},
                     'identity': {
                         'win_sec': 30,
                         'compute_frames': cycle([False] + [True] + [False] * 28),
                         'default_key': 'raw'},
                     'signal_buffer': {'high_rr_quality': 0.7,
                                       'prep': {'to_freq': 10,
                                                'num_taps': -1,
                                                'cutoff': -1}}
                     }
back_chair_config['ra']['win_sec'] = back_chair_config['rr']['win_sec']
back_chair_config['ra']['min_win_sec'] = back_chair_config['rr']['min_win_sec']

front_chair_config = copy.deepcopy(back_chair_config)
front_chair_config['hr']['rest_motion_toggle_parameters']['low_prominence'] = 0.6
front_chair_config['hr']['rest_motion_toggle_parameters']['maximal_derivative_high'] = 100
front_chair_config['hr']['rest_motion_toggle_parameters']['maximal_derivative'] = 60
front_chair_config['rr']['rr_autocorr_parameters']['max_2nd_derivative_zero_crossings'] = 4

front_chair_config['stat']['model_path'] = 'models/rr_svm_status_front.model'
front_chair_config['stat']['model_parameters'] = {'coefficients': np.array([-0.52301993, -1.43596706]),
                                                  'intercept': -2.32943431,
                                                  'scale': np.array([0.2391618,  0.53064701]),
                                                  'mean': np.array([0.56339746, 0.94742249])}
front_chair_config['stat']['motion']['phase_derivative'] = 80
front_chair_config['intra_breath']['phase_derivative_threshold'] = 12
front_chair_config['intra_breath']['filter_degree'] = 1
front_chair_config['range_config']['max_distance'] = 1800
front_chair_config['range_config']['constant_bin'] = None

above_bed_config = copy.deepcopy(front_chair_config)
above_bed_config['posture'] = {'model_path': {'hdf5': 'posture/fmcw_top_1000_bed.hdf5',
                                              'json': 'posture/fmcw_top_1000_bed.json'},
                               'fs': 500,
                               'time_for_update_mean_std': 30,
                               'confidence': 0.5,
                               'compute_frames': cycle([True]),
                               'win_sec': 10,
                               'rel_time_window': 10,
                               'rel_thresh': 8,
                               'prediction_window': 60}

under_bed_config = copy.deepcopy(back_chair_config)

front_standing_config = copy.deepcopy(front_chair_config)
front_standing_config['hr']['rest_motion_toggle_parameters']['maximal_derivative'] = 80
front_standing_config['hr']['rest_motion_toggle_parameters']['low_hr'] = 0     # degenerated
front_standing_config['hr']['rest_motion_toggle_parameters']['max_median_rolling'] = 10000      # degenerated
front_standing_config['hr']['regular_phase_preprocess_quality_advantage'] = -0.2
front_standing_config['hr']['prep']['low'] = 50 / 60
front_standing_config['hr']['prep']['high_filter_pass_method'] = 'filtfilt'
front_standing_config['hr']['folding']['min_hr_to_fold_amplitude_ft'] = 180


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

bioid_config = {
    'classifiers': {
        'linear': {
            'threshold_positive_classified_heartbeats': 0.5},
        'nn': {
            'hdf5': 'bioid_nn_weights.hdf5',
            'json': 'bioid_nn_model.json',
            'meta': 'bioid_nn_metadata.npy'}},
    'input_fs': 500.,  # Hz
    'resampling_fs': 100.,  # Hz
    'downsampling': {
        'fir_downsampling': {
            'taps': 211,
            'cutoff': 50,  # Hz
        },
    },
    'heartbeat': {
        'fir_high_bandpass': {
            'taps': 25,
            'cutoff': [8, 24],  # Hz
        },
        'fir_low_bandpass': {
            'taps': 301,
            'cutoff': [30 / 60, 200 / 60],  # Hz
        },
        'fir_moving_average': {
            'short': 0.4,
            'long': 1.2,
        },
        'fitclip': 5.,
    },
    'bcg': {
        'fir_high_bandpass': {
            'taps': 301,
            'cutoff': [50 / 60, 1200 / 60],  # Hz
        },
        'fir_gain_control_delay': {
            'taps': 601,
        },
        'fir_low_bandpass': {
            'taps': 601,
            'cutoff': 0.4,  # Hz
        },
        'fitclip': 5.,
    },
    'cwt': {
        'default_Omega': 5.0,
        'wavelet': {
            'taps': 80,
        },
        'fir_lowpass': {
            'taps': 601,
            'cutoff': 0.4,  # Hz
        },
        'fir_gain_control_delay': {
            'taps': 601
        },
        'fitclip_low': 0.,
        'fitclip_high': 5.,
        'cwt_freqs': [8., 10., 12., 14., 16., 18.],  # Hz
    },
    'feature_length': 6,  # seconds
    'wait': 10.  # seconds
}

hrv_config = {
    'sdnn': {
        'expected_inlier_fraction': 0.8,  # fraction of intervals expected to be correct
        'outlier_threshold': 20.,  # sample is outlier when outside of the interval mu +- n*signma
    },
    'rmssd': {
        'expected_inlier_fraction': 0.8,  # fraction of intervals expected to be correct
        'outlier_threshold': 20.,  # sample is outlier when outside of the interval mu +- n*signma
    },
}
