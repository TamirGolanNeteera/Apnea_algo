import pandas as pd
import re
import os
import bioid.BioidML as bml
import numpy as np
import pythondev.Tests.vsms_db_api as dbapi
from sklearn.model_selection import train_test_split as trn_tst_split


def iter_matching(dirpath, regexp):
    """ Return all files in dir whose name matches the regular expression."""
    for dir_, dirnames, filenames in os.walk(dirpath):
        for filename in filenames:
            abspath = os.path.join(dir_, filename)
            if regexp.search(filename):
                yield abspath


class BioIDDB:
    def __init__(self, datadir, autofetch=True):
        """
        Keep track of setups in the setup database and with setup 'fingerprints'.
        This enables the easy creation of classifiers, datasets and more.

        Args:
            :param datadir: Directory with results from running a linearBbioID classifier.
            :type datadir: str
            :param autofetch: Default True: autoload the database and results into the object.
            :type autofetch: Bool
        """
        self.datadir = datadir
        self.setupdb = None
        self.iddb = None
        if autofetch:
            self.fetch_setupdb()
            self.fetch_iddb()
        
    def fetch_setupdb(self):
        """
        Fetch the setups in the Neteera database.
        """
        db = dbapi.DB()
        df = pd.DataFrame(db.setup_view())
        df['subject'] = df['subject'].str.strip()  # Remove trailing whitespace of subject name entry in DB.
        self.setupdb = df
        return df

    def fetch_iddb(self):
        """
        Fetch the setups and fingerprints in the fingerprint directory. 
        """
        df = pd.DataFrame(columns=['setup', 'filename', 'fingerprint', 'scores'])
        files = [f for f in iter_matching(self.datadir, re.compile('identity'))]
        for file in files:
            m = re.match("(?P<setup>[0-9]+)_identity_spot\.data", os.path.split(file)[-1])
            if m['setup'] is not None:
                n = int(m['setup'])
                o = bml.loadfingerprint(file)
                s = bml.loadscores(file)
                # assert n == o.getid(), f"Filename setup id {n} does not match fingerprint id {o.getid()}."
                df = df.append({'setup': n, 'filename': file, 'fingerprint': o, 'scores': s}, ignore_index=True)
        self.iddb = df
        return df

    def dataset(self, name):
        """
        Return a named bioid dataset.
        Currently supported:
        - 'sitting_back'
            Description: 'Rest in chair', sitting and measured from the back. Between June 2020 and January 2021.
        - 'all_sitting_back'
            Description: 'Rest in chair', sitting and measured from the back. From June 1, 2020.
        """
        assert name in self.supported_datasets(), "Unknown Bio-ID dataset name."
        if self.setupdb is None:
            df = self.fetch_setupdb()
        else:
            df = self.setupdb

        if name == 'sitting_back':
            dg = df[
                (df['timestamp'] > '2020-06-01 00:00:00') &  # filter out older sessions from before June 1, 2020
                (df['timestamp'] < '2021-01-17 09:00:00') &  # filter out newer sessions from after January 17, 2021
                (df['mode'] == 'FMCW') &  # only FMCW
                (df['target'] == 'back') &
                (df['validity'].isin(['Valid', 'Confirmed'])) &
                (df['scenario'].isin(['Rest in chair', 'Mild motion, Rest in chair'])) &
                (df['duration'] > 30) &
                (df['duration'] < 800) &
                (df['distance'] < 200) &
                (df['posture'] == 'Sitting') &
                (df['location'] == 'seat_back') &
                (df['company'] == 'Neteera') &
                (df['notes'] != 'FFT32 test')
            ]
            return dg
        elif name == 'all_sitting_back':
            dg = df[
                (df['timestamp'] > '2020-06-01 00:00:00') &  # filter out older sessions from before June 1, 2020
                (df['mode'] == 'FMCW') &  # only FMCW
                (df['target'] == 'back') &
                (df['validity'].isin(['Valid', 'Confirmed'])) &
                (df['scenario'].isin(['Rest in chair', 'Mild motion, Rest in chair'])) &
                (df['duration'] > 30) &
                (df['duration'] < 800) &
                (df['distance'] < 200) &
                (df['posture'] == 'Sitting') &
                (df['location'] == 'seat_back') &
                (df['company'].isin(['Neteera', 'SZMC'])) &
                (df['notes'] != 'FFT32 test')
                ]
            return dg
        elif name == '10_setups_each':
            # return the 10 latest setups for all 11 people that had 10 sessions until 21-02-2021
            dg = df[
                (df['timestamp'] > '2020-06-01 00:00:00') &  # filter out older sessions from before June 1, 2020
                (df['timestamp'] < '2021-02-21 00:00:00') &  # filter out older sessions from before June 1, 2020
                (df['mode'] == 'FMCW') &  # only FMCW
                (df['target'] == 'back') &
                (df['validity'].isin(['Valid', 'Confirmed'])) &
                (df['scenario'].isin(['Rest in chair', 'Mild motion, Rest in chair'])) &
                (df['duration'] > 30) &
                (df['duration'] < 800) &
                (df['distance'] < 200) &
                (df['posture'] == 'Sitting') &
                (df['location'] == 'seat_back') &
                (df['company'].isin(['Neteera'])) &
                (df['notes'] != 'FFT32 test')
                ]

            db = dbapi.DB()
            dg = dg.assign(fs=[db.setup_fs(idx) for idx in dg.setup])
            dg = dg[dg['fs'] == 500.]
            dg = (dg.sort_values('timestamp', ascending=False)\
                    .groupby('subject')\
                    .filter(lambda x: len(x['setup']) >= 10)\
                    .groupby('subject')\
                    .head(10))
            return dg
        else:
            return None

    def supported_datasets(self):
        """ Return the names of the datasets that are available in the function self.dataset(name). """
        return ['sitting_back', 'all_sitting_back', '10_setups_each']

    def getstats(self, idx):
        i = set(idx).difference(set(self.iddb['setup']))
        assert len(i) == 0, f"Not all requested setups are available: {i}"
        s = self.iddb[self.iddb['setup'].isin(idx)].fingerprint.values
        return s

    def one_v_all(self, df, balanced=True):
        """
        Compute one_v_all classifiers for the unique subjects in DataFrame df.

        Args:
            :param df: Dataframe with the setups requested for the classifiers
            :type: DataFrame
            :param balanced: Balance classes
            :type: Bool
        """
        missingprints = set(df.setup).difference(set(self.iddb.setup))
        if len(missingprints) > 0:
            print(f"Missing fingerprints for setups {missingprints}.")

        c = []  # classifiers
        for s in df.subject.unique():
            try:
                dsu = df[(df.subject == s) & (~df.setup.isin(missingprints))]  # subject
                dns = df[(df.subject != s) & (~df.setup.isin(missingprints))]  # everybody else
                assert dsu.shape[0] >= 1, f'No fingerprints for {s}'
                assert dns.shape[0] >= 1, f'No fingerprints for people other than {s}'
                osu = [x for x in self.getstats(dsu.setup) if x]
                ons = [x for x in self.getstats(dns.setup) if x]
                assert len(osu) >= 1, f'No fingerprints for {s}'
                assert len(ons) >= 1, f'No fingerprints for people other than {s}'
                pos = bml.merge_stats(osu)
                neg = bml.merge_stats(ons)
                c.append(bml.classifier(pos, neg, f'{s}_v_all', balanced=balanced, labels={True: s, False: f'Not {s}'}))
            except AssertionError as e:
                print(e)
        return c

    def one_v_all_results(self, classifiers=None, dataset='test'):
        assert dataset in ['train', 'test', 'all'], "Choose train, test, or all for dataset."
        results = pd.merge(self.setupdb, self.iddb, on='setup')
        g = []
        for r in results.itertuples():
            if r.scores is not None:
                d = pd.DataFrame.from_records(r.scores.copy())
                d['setup'] = r.setup
                d['subject'] = r.subject
                d['train'] = np.any([r.setup in p for p in classifiers.precursors])
                g.append(d)
        results = pd.concat(g)

        if dataset == 'train':
            subset = results[results['train']]
        elif dataset == 'test':
            subset = results[~results['train']]
        else:
            subset = results
        subset = subset.drop(columns='train')
        dg = subset.\
            pivot(index=['subject', 'setup'], columns=['id']).\
            groupby('subject').\
            mean().\
            reset_index()

        cols = np.asarray(dg.columns.levels[1].values[:-1])

        rows = dg.iloc[:, 0].to_numpy()

        cols_for_subjects = [True for _ in cols]

        R = dg.iloc[:, 1:].to_numpy()
        return R[:, cols_for_subjects], cols[cols_for_subjects], rows


def train_test_split(df, ratio=0.7, seed=1337):
    """ Split a dataframe into training and testing with a ratio that holds equally for every subject.

    Args:
        :param df: DataFrame with setup numbers and subjects.
        :type df: DataFrame
        :param ratio: ratio of the size of the training set compared to the number of setups in df
        :type ratio: float
        :param seed: seed value for the sklearn train_test_split function.
        :type seed: int
    """
    trn = []
    tst = []
    for (subject, setups) in df.groupby('subject'):
        if setups.shape[0] > 1:
            gtrn, gtst = trn_tst_split(setups, train_size=ratio, random_state=seed, shuffle=True)
            trn.append(gtrn)
            tst.append(gtst)
        else:
            trn.append(setups)
    return pd.concat(trn), pd.concat(tst)


def one_v_all_performance(result_matrix):
    n = result_matrix.shape[0]
    true_positives = np.diag(result_matrix)  # on average, how often each person's classifier classified
    # their person correctly, higher is better
    false_positives = result_matrix - np.diag(np.diag(result_matrix))  # how often someone else
    # was classified as the specific person, lower is better


    return {
        'mean_true_positives': np.mean(true_positives),
        'mean_false_positives': np.sum(false_positives) / (n * (n - 1)),
        'min_true_positives': np.min(true_positives),
        'max_false_positives': np.max(false_positives),
    }