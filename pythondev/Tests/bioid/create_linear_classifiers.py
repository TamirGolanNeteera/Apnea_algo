import Tests.bioid.BioidDB as bdb
import bioid.BioidML as bml
fpdir = '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/fingerprints/v10_fingerprints'
cldir = '/Neteera/Work/homes/reinier.doelman/Documents/projects/results/bioid/classifiers/v11_identity'

db = bdb.BioIDDB(fpdir)
df = db.dataset('all_sitting_back')
df = df[df.company == 'Neteera']
df = df[~df.setup.isin([
    5492, 5494, 4041, 5159, 5488, 5490,
    4560, 4562, 5240, 5242, 5412, 5414,
    5516, 5518, 5536, 5538, 5445, 5447])]
df_train, df_test = bdb.train_test_split(df, ratio=0.8)
c = db.one_v_all(df)

# [f'{x}' for x in df.setup]
for ci in c:
    bml.save_regression(cldir, ci, ci.meta['labels'][True])

# Test set sessions for different people
# df_test[df_test.subject.isin(['Ohad Basha', 'Shahar Yaron', 'Reinier Doelman', 'Hanna Riez', 'Idan Yona', 'David Grossman', 'David Dadoune'])].setup.values
# df_train[df_train.subject.isin(['Ohad Basha', 'Shahar Yaron', 'Reinier Doelman', 'Hanna Riez', 'Idan Yona', 'David Grossman', 'David Dadoune'])].setup.values