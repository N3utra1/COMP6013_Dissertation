import mne
from mne.time_frequency import stft, stftfreq
import os
import pandas as pd
import threading
import control
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm


class Extractor:
    # Extracts the infomration from each EDF_File contained in a CHB Object
    def __init__(self, chb_metadata=None, csv_path=None, write_path=None, overwrite=False, threading=False):
        self.chb_metadata = chb_metadata
        self.csv_path = csv_path 
        self.write_path = write_path
        self.overwrite = overwrite
        if threading:
            self.extract_threaded() 
        else: self.extract() 

    def extract_threaded(self):
        print("threading is not implimented, falling back to single threaded extraction")
        self.extract()

    def extract(self):
        for chb_file in self.chb_metadata.edf_files:
            # if no seizures exist skip
            if chb_file.seizure_dict == {}: continue 


            csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]

            ictal_master = pd.read_csv(os.path.join(csv_prefix, "ictal", "master.csv"), header=0)
            preictal_master = pd.read_csv(os.path.join(csv_prefix, "preictal", "master.csv"), header=0)
            interictal_master = pd.read_csv(os.path.join(csv_prefix, "interictal", "master.csv"), header=0)

            dataframes = {'ictal_master': ictal_master, 'preictal_master': preictal_master, 'interictal_master': interictal_master}
            largest_df_name = max(dataframes, key=lambda x: dataframes[x].size)
            largest = dataframes[largest_df_name]

            target_number_of_windows = largest.size // (control.window_size * chb_file.data_sampling_rate)

            stft_prefix = os.path.splitext(os.path.join(control.stft_extraction_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
            self.generate_windows(ictal_master, target_number_of_windows, chb_file.data_sampling_rate, os.path.join(stft_prefix, "ictal"))
            self.generate_windows(preictal_master, target_number_of_windows, chb_file.data_sampling_rate, os.path.join(stft_prefix, "preictal"))
            self.generate_windows(interictal_master, target_number_of_windows, chb_file.data_sampling_rate, os.path.join(stft_prefix, "interictal"))


    def generate_windows(self, df, target_number, sampling_rate, write_path_prefix):
        start = 0
        end = df.size
        window_size = (control.window_size * sampling_rate)

        if window_size > end: return
        file, mode = os.path.split(write_path_prefix)[-2:]
        print(f"writing {mode} for {file}")
        step = (end - start - window_size) / (target_number- 1)
        windows = [(int(i * step), int(i * step + window_size)) for i in range(target_number)]

        for window in windows:
            w_start = window[0]
            w_end = window[1]
            df_window_slice = df.iloc[w_start : w_end]
            info = mne.create_info(ch_names=control.common_columns, sfreq=sampling_rate)
            raw = mne.io.RawArray(df_window_slice[control.common_columns].transpose(), info, verbose=False)
            stft_data = stft(raw.get_data(), window_size, verbose=False)
            output_path = os.path.join(write_path_prefix, f"{w_start}-{w_end}-stft.npy")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            np.save(output_path, stft_data)