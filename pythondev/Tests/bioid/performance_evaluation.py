import os
import numpy as np
import re
import pandas as pd
import pythondev.Tests.vsms_db_api as dbapi
from collections import Counter

def iter_matching(dirpath, regexp):
    for dir_, dirnames, filenames in os.walk(dirpath):
        for filename in filenames:
            abspath = os.path.join(dir_, filename)
            if regexp.search(filename):
                yield abspath

result_dirs = {
    'spot': {
        'nn': '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/fingerprints/v1_nn_performance_spot',
        'linear': '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/fingerprints/v2_linear_performance_spot',

    },
    'continuous': {
        'nn': '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/fingerprints/v1_nn_performance_continuous',
        'linear': '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/fingerprints/v2_linear_performance_continuous'
    }
}

regexs = {
    'continuous': re.compile('(?P<idx>[0-9]{4,})_identity.npy'),    # continuous
    'spot': re.compile('(?P<idx>[0-9]{4,})_identity_spot.data'),    # spot
}

method = 'continuous'  # spot / continuous
classifier = 'linear'  # nn / linear
rgx = regexs[method]
dir = result_dirs[method][classifier]

files = [f for f in iter_matching(dir, rgx)]

db = dbapi.DB()
results = []
for f in files:
    m = rgx.search(f)
    idx = int(m.group('idx'))
    subject = db.setup_subject(idx)
    result = np.load(f, allow_pickle=True)
    results.append({'idx': idx, 'filename': f, 'subject': subject, 'result': result})

df_results = pd.DataFrame(results)

predictions = []
if method == 'continuous':
    for index, row in df_results.iterrows():
        pred_identity = [x['identity'] for x in row.result if x['identity'] != 'Collecting data...' and x['identity'] != 'No match']
        no_pred_identity = [x['identity'] for x in row.result if x['identity'] == 'No match']
        number_predictions = len(pred_identity)
        number_no_predictions = len(no_pred_identity)
        counts = dict(Counter(pred_identity))
        counts_fraction = {k: counts[k]/number_predictions for k in counts.keys()}
        if row.subject in counts.keys():
            correct = counts[row.subject]
        else:
            correct = 0.
        counts.update({
            'idx': row.idx,
            'subject': row.subject,
            'counts': counts,
            'counts_fraction': counts_fraction,
            'correct': correct,
            'number_predictions': number_predictions,
            'number_no_predictions': number_no_predictions,
        })
        predictions.append(counts)
    df_predictions = pd.DataFrame(predictions)
    no_predict = np.sum(df_predictions.number_no_predictions)
else:
    for index, row in df_results.iterrows():
        pred_identity = row.result['identity']
        predictions.append({
            'identity': pred_identity,
            'idx': row.idx,
            'subject': row.subject,
            'counts': 1,
            'counts_fraction': 1*(row.subject == pred_identity),
            'correct': 1*(row.subject == pred_identity),
            'number_predictions': 1,
        })
    df_predictions = pd.DataFrame([p for p in predictions if p['identity'] != 'No match'])
    df_nopredictions = pd.DataFrame([p for p in predictions if p['identity'] == 'No match'])
    no_predict = df_nopredictions.shape[0]
df_predictions = df_predictions.fillna(0)
# classes = df_predictions.columns.difference(['idx', 'subject', 'counts', 'counts_fraction', 'correct', 'number_predictions'])
# df_predictions[classes]


correct = sum(df_predictions['correct'])
predictions = sum(df_predictions['number_predictions'])
# no_predict = df_nopredictions.shape[0]

print(f'Algo: {method}. Classifier: {classifier}')
print(f'Correct predictions: {correct/predictions}')
print(f'No predictions: {no_predict/(no_predict + predictions)}')
