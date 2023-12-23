import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
base_path = '/Neteera/Work/homes/dana.shavit/Research/BP2023/csv/'

ranges_dict = {'low':[], 'mid':[], 'high':[]}
setup_files = glob.glob(base_path+'/*.csv')#fnmatch.filter(base_path, '*.csv')
for f in setup_files:
    param_file = pd.read_csv(f)
    print(f)
    print(len(param_file))
    fig, ax = plt.subplots(2, sharex=False, figsize=(20,10))
    ax[0].set_title(f[52:])
    valid_time_stamps = []
    valid_s_measurements = []
    try:
        ax[0].plot(param_file[' Art-S(mmHg)'], linewidth=0.5)
        art_s = param_file[' Art-S(mmHg)'].to_numpy()
        for i in range(len(art_s)):
            if art_s[i]=='--':
                art_s[i] = -1
            else:
                art_s[i] = int(art_s[i])
        ax[1].plot(art_s, linewidth=0.5)
    except:
        print(1)
        continue

    param_file[' Art-S(mmHg)'] = art_s
    value_ranges = [-2, 30,100,120,130,145,200, 1000]


    # Create labels for each segment
    segment_labels = ['-1', 'low', 'ignore1', 'mid', 'ignore2', 'high', 'ignore_high']

    # Use pd.cut() to create segments
    param_file['segment'] = pd.cut(param_file[' Art-S(mmHg)'], bins=value_ranges, labels=segment_labels)

    # Display the DataFrame with segments

    for segment, segment_data in param_file.groupby('segment'):

        if segment in ['low', 'mid', 'high'] and len(segment_data):
            ranges_dict[segment].append([  f[f.rfind('/')+1:], segment_data.head(1)['Time'].to_numpy()[0], segment_data.tail(1)['Time'].to_numpy()[0]])

    ax[1].axhline(y = 145, linewidth=0.5, c='red')
    ax[1].axhline(y = 100, linewidth=0.5, c='blue')
    plt.savefig(f[:-3]+'png')
    #plt.show()
    plt.close()
import json
print(ranges_dict)
with open(os.path.join(base_path,'ranges_dict.json'), 'w') as json_file:
    json.dump(ranges_dict, json_file)