import numpy as np
import scipy.signal as sp
import datetime as dt
TRUE_POSITIVE_TOLERANCE_MS = 15
import beatdetection.BeatDetection as bd

class Heartbeats:
    def __init__(self, r_peaks, start):
        self.r_peaks = [int(r) for r in r_peaks]  # milliseconds after start of recording where r-peaks are estimated or detected
        self.start = start
        try:
            self.end = self.start + dt.timedelta(milliseconds=int(r_peaks[-1]))
        except IndexError as e:
            print(e)

    def crop(self, start, end):
        """ Crop Heartbeat object to start and end time """
        new_start = max(start, self.start)
        new_end = min(end, self.end)
        millisecond_drop_start = int(round((new_start - self.start) / dt.timedelta(milliseconds=1)))
        millisecond_drop_end = int(round((self.end - new_end) / dt.timedelta(milliseconds=1)))
        self.r_peaks = [r - millisecond_drop_start for r in self.r_peaks if (r > millisecond_drop_start and r < self.r_peaks[-1] - millisecond_drop_end + 1) if r - millisecond_drop_start > 0]
        self.start = new_start
        self.end = new_start + dt.timedelta(milliseconds=self.r_peaks[-1])


def crop(h1, h2):
    """ Cut two objects to the same start and end time for full overlap"""
    start = max(h1.start, h2.start)
    end = min(h1.end, h2.end)
    h1.crop(start, end)
    h2.crop(start, end)


def finddelay(x, y):
    s = sp.correlate(y, x, mode='full')
    return x.shape[0] - np.argmax(s) - 1


def finddelay_peaks(peaks_a, peaks_b, width=8):
    """ Find delay between two lists of peaks. """
    va = np.zeros(peaks_a[-1])
    vb = np.zeros(peaks_b[-1])
    for i in peaks_a:
        va[i-1] = 1
    for i in peaks_b:
        vb[i-1] = 1
    vaf = sp.filtfilt(np.ones(width), [1], va)
    vbf = sp.filtfilt(np.ones(width), [1], vb)
    return finddelay(vaf, vbf)


def performance(heartbeats_ref, heartbeats_est, true_positive_tolerance=[TRUE_POSITIVE_TOLERANCE_MS]):
    crop(heartbeats_ref, heartbeats_est)  # trim away ends by time stamps
    offset_n = finddelay_peaks(heartbeats_ref.r_peaks, heartbeats_est.r_peaks)
    heartbeats_est_delayed = Heartbeats(heartbeats_est.r_peaks + offset_n, heartbeats_est.start)
    crop(heartbeats_ref, heartbeats_est_delayed)  # trim to overlapping part
    results = compare_heartbeats(heartbeats_ref, heartbeats_est_delayed, true_positive_tolerance)
    results['estimated offset milli-sec'] = offset_n
    hrv_est = bd.hrv(2 * np.diff(heartbeats_est.r_peaks) / 1000)[0]
    hrv_ref = bd.hrv(2 * np.diff(heartbeats_ref.r_peaks) / 1000)[0]
    results['hrv est sdnn'] = hrv_est['sdnn']
    results['hrv est rmssd'] = hrv_est['rmssd']
    results['hrv ref sdnn'] = hrv_ref['sdnn']
    results['hrv ref rmssd'] = hrv_ref['rmssd']
    return results


def compare_heartbeats(peaks_ref, peaks_est, tolerance_list):
    output = {}
    r_ref = np.asarray(peaks_ref.r_peaks)
    r_est = np.asarray(peaks_est.r_peaks)
    n_peaks_reference = len(r_ref)
    n_peaks_estimated = len(r_est)

    # Find estimated peaks within tolerance of reference peaks
    for tolerance in tolerance_list:
        tp_matches = [(i, np.nonzero(np.abs(s - r_est) < tolerance)[0]) for i, s in enumerate(r_ref)]
        true_positives = len([s for s in tp_matches if len(s[-1]) >= 1])
        false_negatives = len([s for s in tp_matches if len(s[-1]) == 0])

        fp_matches = [(i, np.nonzero(np.abs(s - r_ref) < tolerance)[0]) for (i, s) in enumerate(r_est)]
        false_positives = len([s for s in fp_matches if len(s[-1]) == 0])
        output[f'true positives tolerance {tolerance} ms'] = true_positives
        output[f'false positives tolerance {tolerance} ms'] = false_positives
        output[f'false negatives tolerance {tolerance} ms'] = false_negatives
        output[f'true positive rate tolerance {tolerance} ms'] = true_positives / n_peaks_reference
    deviation = np.asarray([r_ref[i] - r_est[j[-1]] for (i, j) in tp_matches if np.shape(j)[0] == 1])

    output.update({'n peaks reference': n_peaks_reference,
                   'n peaks estimated': n_peaks_estimated,
                   'mean absolute deviation': np.mean(np.abs(deviation))})

    return output

