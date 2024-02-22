import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import seaborn as sns  # for enhanced styling


from Tests.NN.create_apnea_count_AHI_data import NW_HQ, MB_HQ
from Tests.Utils.LoadingAPI import load_reference
from Tests.vsms_db_api import DB, Sensor

Raligh_setups = [112091, 112092, 112093, 112094, 112199, 112200, 112201, 112202, 112036, 112037, 112038, 112039, 112095, 112096, 112097, 112098, 112040, 112041, 112042, 112043, 112102, 112103, 112104, 112105, 112989, 112990, 112991, 112992, 112993, 112994, 112195, 112196, 112197, 112198]
Sumter_setups = [112297, 112298, 112299, 112300, 112301, 112359, 112360, 112361, 112362, 112363, 112364, 112365, 112366, 112116, 112117, 112118, 112119, 112120, 112121, 112122, 112123, 112150, 112151, 112152, 112153, 112154, 112155, 112135, 112136, 112137, 112138, 112139, 112140, 112141, 112051, 112052, 112053, 112054, 112055, 112056, 112427, 112428, 112429, 112430, 112431, 112432, 112433, 112434, 113011, 113012, 113013, 113014, 113015, 113016, 113017, 112392, 112393, 112394, 112395, 112396, 112397, 112057, 112058, 112059, 112060, 112061, 112062, 112166, 112167, 112168, 112169, 112170, 112024, 112025, 112026, 112027, 112028, 112029, 112030, 112031, 112343, 112344, 112345, 112346, 112347, 112348, 112007, 112008, 112009, 112010, 112011, 112012, 112013, 112014, 112015, 112128, 112129, 112130, 112131, 112132, 112133, 112134, 113006, 113007, 113008, 113009, 113010, 112418, 112419, 112420, 112421, 112422, 112423, 112424, 112425, 112205, 112206, 112207, 112208, 112209, 112210, 112211, 112212, 112213, 112214, 112215, 112384, 112385, 112386, 112387, 112388, 112389, 112390, 112391, 113018, 113019, 113020, 113021, 113022, 112305, 112306, 112307, 112308]


setups = MB_HQ + NW_HQ + Raligh_setups + Sumter_setups
setups = [s for s  in setups if s not in [109884, 109886, 110393, 110394]]
if __name__ == '__main__':
        db = DB()
        ss_dict = {}
        subjects = set()
        for setup in setups:
                db.update_mysql_db(setup)
                session = db.session_from_setup(setup)
                subjects.add(db.setup_subject(setup))
                if session in ss_dict.keys():
                        continue
                ss = load_reference(setup, 'sleep_stages', db)
                if isinstance(ss, pd.Series):
                        ss = ss.to_numpy()
                if 'W' not in ss:
                        print(f'setup {setup} has no "W" in it')
                ss[pd.isna(ss)] = 'W'
                ss[ss=='invalid'] = 'W'
                ss[ss=='?'] = 'W'
                if len(set(np.unique(ss)) -  {'W', 'N1', 'N2', 'N3', 'R', 'REM'}):
                        print('values error')
                sleep_time = len(np.where(ss != 'W')[0])/3600
                ss_dict[session] = sleep_time
        sleep_times = list(ss_dict.values())


        # Identify sessions with different sleep duration ranges
        # Define bins with half-hour intervals
        bins = np.arange(2, 10.5, 0.5)
        bin_labels = [f'{bin_left} to {bin_right}' for bin_left, bin_right in zip(bins, bins[1:])]

        # Calculate sleep counts in each bin
        sleep_counts = [sum(1 for time in sleep_times if bin_left <= time < bin_right) for bin_left, bin_right in zip(bins, bins[1:])]

        # Set seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Plot histogram with enhanced styling
        plt.figure(figsize=(10, 6))
        sns.histplot(sleep_times, bins=bins, kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Overall Sleep Time Distribution of {len(ss_dict)} sessions', fontsize=16)

        # Display the counts in each bin
        for bin_left, bin_right, count in zip(bins, bins[1:], sleep_counts):
            plt.text((bin_left + bin_right) / 2, 0.5, f'{count}', ha='center', va='bottom', fontsize=10, color='red')

        # Set x-axis labels
        # plt.xticks(bins, bin_labels, rotation=45, fontsize=12)
        plt.xlabel(f'Sleep Time (hours)', fontsize=14)

        # Set y-axis labels as integers
        plt.yticks(np.arange(0, max(sleep_counts) + 1, 1), fontsize=12)
        plt.ylabel('Frequency', fontsize=14)

        # Add information about counts between 2 and 2.5, and between 2.5 and 3 in the title


        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()