#   general:
import platform

UNITS = {'hr': 'bpm', 'rr': 'bpm', 'bbi': 'ms', 'ra': 'microns', 'ie': '', 'spo2': '%'}

STATUS_TO_CLASS = {'empty chair': 0, 'running': 1, 'breath holding': 2, 'empty': 0, 'low respiration': 2,
                   'warming up': -1, 'motion': 3, 'disabled': -1}

STAT_CLASSES = ['empty', 'normal', 'zrr', 'motion']

#   for evaluation:
ULTRA_CATEGORIES = ['Rest', 'Motion', 'Driving', 'Back', 'Chest', 'Above Bed', 'Under Bed', 'Standing', 'Chair Back',
                    'Chair Front', 'Bed', 'lying on back']

BBI_TOLERANCE = [15, 30, 50, 100]  # ms

UNDER_DICT = {'hr': [{'per': 10, 'diff': 0.05},  # < 10% error
                     {'thresh': 5, 'diff': 0.1},
                     {'thresh': 6, 'diff': 0.2},
                     {'thresh': 7, 'diff': 0.2},
                     {'thresh': 8, 'diff': 0.2},
                     {'thresh': 5, 'per': 4, 'diff': 0.2},
                     {'thresh': 5, 'per': 5, 'diff': 0.2},
                     {'per': 5, 'diff': 0.1},
                     # {'thresh': 20, 'diff': 0.001},  # < 10 bpm error
                     # {'thresh': 40, 'diff': 0.001},
                     ],
              'rr': [
                  {'per': 10, 'thresh': 2, 'diff': 0.05},
                  {'thresh': 4, 'diff': 0.01},
                  {'thresh': 3, 'diff': 1},
                  {'thresh': 10, 'diff': 0.001},
              ],
              'ie': [{'per': 20},
                     {'thresh': 0.5},
                     {'thresh': 4}]}

VERY_LOW_VALUES = {'hr': 50, 'rr': 6}
VERY_HIGH_VALUES = {'hr': 100, 'rr': 27}

VITAL_SIGN_LIMITS = {'rr': {'min': 2, 'max': 60},
                     'hr': {'min': 30, 'max': 300},
                     'bbi': {'min': 400, 'max': 2000},
                     'ie': {'min': 0.01, 'max': 2000}}

PER_SETUP_THRESHOLDS = [90, 95]  # in percentage
PER_SETUP_THRESHOLDS = []

if platform.system() == "Linux":
    DELIVERED = '/Neteera/Work/NeteeraVirtualServer/DELIVERED/Algo/'
else:
    DELIVERED = r'N:\NeteeraVirtualServer\DELIVERED\Algo'

CATEGORIES = {'sleep_stages': ['W', 'N1', 'N2', 'N3', 'R'],
              'apnea': ['normal', 'Hypopnea', 'Central', 'Obstructive', 'Mixed', 'Apnea']}
