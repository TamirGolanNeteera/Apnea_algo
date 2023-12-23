from scipy.signal import butter, lfilter
from Tests.vsms_db_api import *
from Tests.NN.create_apnea_count_AHI_data import load_radar_data, load_radar_bins, radar_cpx_file, compute_phase, compute_respiration
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

home = '/Neteera/Work/homes/dana.shavit/Research/210/'

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import signal
    import scipy as sp
    import beatdetection.BeatDetectionDSP as bdd
    from Tests.Utils.LoadingAPI import load_reference
    import glob
    bin_qual_list = []
    setup1 = db.all_setups()

    setup2 = []
    for s in setup1:
        if s > 10000:
            if 'FDA' in db.setup_note(s) and db.setup_target(s) =='front' and db.setup_distance(s) > 400:
            ##if 'FDA chair rest protocol-1' in db.setup_note(s) or 'Empty be protocol-1' in db.setup_note(s):#db.setup_target(s) =='front' and db.setup_distance(s) > 400 and
                setup2.append(s)

    setup2 = np.stack(setup2)
    setup3 = []
    for s in setup2:
        print(s, db.setup_sn(s), db.setup_duration(s))
        if db.setup_sn(s):
            if db.setup_sn(s)[0][0] == '2':
                setup3.append(s)
    print(setup2)
    flist =  glob.glob(home+'**/*quality_phase*.npy', recursive=True)
    res_dict = []
    res = []

    if True:
        for f in flist:
            bin = int(f[f.find('FD_')+3:f.find('accumulated')-1])

            setup = int(f[f.find('accumulated')+len('accumulated')+1:f.find('_quality')])

            qual = np.load(f, allow_pickle=True)
            #print(setup, bin, np.median(qual))
            bin_qual_list.append(np.median(qual))
            res_dict.append([setup, bin, np.median(qual)])

        res_dict =np.stack(res_dict)
        setups = np.unique(res_dict[:,0])

        for j in setups:
            fig, ax = plt.subplots(8, sharex=False)

            print(j)
            db.update_mysql_db(j)

            #print(j, db.setup_target(j))
            p = db.setup_dir(j)
            radar_file = radar_cpx_file(j)
            cpx, setup_fs = load_radar_data(radar_file)

            bins = load_radar_bins(radar_file)
            rng_var = np.var(bins, axis=0)
            selected_bin = np.argmax(rng_var)

            ax[7].set_title(str(j) + ' median quality by bin', fontsize=6)
            d = res_dict[res_dict[:, 0] == j]
            d = d[d[:, 1].argsort()]

            col = 'red'
            best_qual_bin = d[:, 1][np.argmax(d[:, 2])]

            best_qual_score = np.max(d[:, 2])

            if  best_qual_bin == selected_bin:
                col = 'blue'

            ax[7].plot(d[:, 1], d[:, 2], color=col)
            ax[7].set_xlabel('bin')
            ax[7].set_ylabel('median quality')

            sn = db.setup_sn(j)[0][0:2] if db.setup_sn(j) else ''
            res.append([db.session_from_setup(int(j)), j, int(sn + '0'), db.setup_distance(j), db.setup_posture(j) , np.max(d[:, 2])])


            sbin = 5#np.max([0, selected_bin-2])
            ebin = 12#np.min([bins.shape[1]-1, selected_bin+3])
            for b in range(sbin, ebin):
                #ax[b-(sbin)].set_title(str(b))
                r = []
                cpx = bins[:, b]
                i = cpx.real
                q = cpx.imag

                phase = compute_phase(cpx)
                phase = compute_respiration(phase)
                i = compute_respiration(cpx.real)
                q = compute_respiration(cpx.imag)
                #print(j, (len(phase)/setup_fs)/60)
                col = 'black'
                lbl=''
                if b == selected_bin:
                    col = 'green'
                    lbl = 'selected '
                elif b == best_qual_bin:
                    col = 'magenta'
                    lbl='highest qual '
                ax[b-sbin].plot(phase, label=lbl+"Q bin "+str(b), color=col, linewidth=0.5)
                #ax[b-sbin].text(10, 10, str(d[b-sbin, 2]), bbox=dict(facecolor='yellow', alpha=0.5))
                if b < ebin-1:
                    ax[b-sbin].tick_params(bottom=False)
                ax[b-sbin].legend(loc='upper right', fontsize=6)

            # bbb = np.stack(bin_qual_list)
            # print(bbb)
            # print(np.median(bbb))
            sn = db.setup_sn(j)[0][0:2] if db.setup_sn(j) else ''
            ax[0].set_title(str(db.session_from_setup(int(j)))+' '+str(int(j)) + ' :: ' + sn + '0 :: '+str(db.setup_distance(j)) + ' :: '+db.setup_posture(j))

            #plt.show()

            plt.savefig(home+str(db.session_from_setup(int(j)))+'_'+str(j)+'_'+str(db.setup_distance(j)) +'_'+ sn + '0_'+db.setup_posture(j)+'.png', dpi=300)
            plt.close()

        import pandas as pd
        p = pd.DataFrame(res)
        p.to_csv(home+'out.csv')
    p = pd.read_csv(home+'out.csv')
    medians130_s = []
    medians210_s = []
    medians130_l = []
    medians210_l = []
    dfSitting = p[p['4'] == 'Sitting']
    dfLying = p[p['4'] == 'Lying']
    results = []#pd.DataFrame(columns=['session', 'setup_130', 'setup_210', 'distance', 'posture', 'diff'])
    for sess in np.unique(dfSitting['0'].to_numpy()):
        ds = dfSitting[dfSitting['0']==sess]
        if len(ds)<2:
            continue
        d130 = ds[ds['2']==130]
        if len(d130) != 1:
            continue
        dist = int(d130['3'])
        setups_at_d = ds[ds['3']==dist]
        if len(setups_at_d) != 2:
            continue
        l210 = setups_at_d[setups_at_d['2']==210]
        l130 = setups_at_d[setups_at_d['2']==130]


        medians210_s.append(l210['5'])
        medians130_s.append(l130['5'])

        results.append([l210['0'].to_numpy()[0], int(l130['1'].to_numpy()[0]), int(l210['1'].to_numpy()[0]), l210['3'].to_numpy()[0], l210['4'].to_numpy()[0], l130['5'].to_numpy()[0] - l210['5'].to_numpy()[0]])
    for sess in np.unique(dfLying['0'].to_numpy()):
        ds = dfLying[dfLying['0']==sess]
        if len(ds)<2:
            continue
        d130 = ds[ds['2']==130]
        if len(d130) != 1:
            continue
        dist = int(d130['3'])
        setups_at_d = ds[ds['3']==dist]
        if len(setups_at_d) != 2:
            continue
        l210 = setups_at_d[setups_at_d['2']==210]
        l130 = setups_at_d[setups_at_d['2']==130]
        medians210_l.append(l210['5'])
        medians130_l.append(l130['5'])
        results.append([l210['0'].to_numpy()[0], int(l130['1'].to_numpy()[0]),int(l210['1'].to_numpy()[0]), l210['3'].to_numpy()[0], l210['4'].to_numpy()[0], l130['5'].to_numpy()[0] - l210['5'].to_numpy()[0]])

    plt.plot(medians130_l, label='130 Lying')
    plt.plot(medians210_l, label='130 Lying')
    plt.legend()
    plt.figure()
    plt.plot(medians130_s, label='130 Sitting')
    plt.plot(medians210_s, label='210 Sitting')
    plt.legend()
    print("Sitting, 130, 210", np.median(medians130_s), np.median(medians210_s))
    print("Lying, 130, 210", np.median(medians130_l), np.median(medians210_l))
    plt.show()
    print("done")
