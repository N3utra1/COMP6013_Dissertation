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
        if self.chb_metadata.name == "chb01": return
        if threading:
            self.extract_threaded() 
        else: self.extract() 

    def extract_threaded(self):
        print("threading is not implimented, falling back to single threaded extraction")
        self.extract()


    def extract(self):
        def count_lines(filename):
            with open(filename, 'r') as f:
                for i, line in enumerate(f, 1):
                    pass
            return i - 1

        def get_class_size(class_type):
            total = 0
            for chb_file in self.chb_metadata.edf_files:
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                file_path = os.path.join(csv_prefix, class_type, "master.csv")
                if os.path.exists(file_path):
                    total += count_lines(file_path)
            print(f"{class_type}  :  {total}")
            return total

        def generate_windows_for_class(class_type):
            all_dfs = []
            for chb_file in self.chb_metadata.edf_files:
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                path = os.path.join(csv_prefix, class_type, "master.csv")
                if os.path.exists(path): all_dfs.append(pd.read_csv(path))
            whole_df = pd.concat(all_dfs) 
            self.generate_windows(whole_df, target_number_of_windows, 256, os.path.join(stft_prefix, class_type))

        dataframes = {'ictal': get_class_size("ictal"), 
                        'preictal': get_class_size("preictal"), 
                        'interictal': get_class_size("interictal")}
        # dataframes = {'ictal': 1, 'preictal': 1, 'interictal': 2} # this forces it to always think interictal is the largest class
        largest_df_name = max(dataframes, key=lambda x: dataframes[x])
        largest = dataframes[largest_df_name]
        target_number_of_windows = largest // (control.window_size * 256)
        stft_prefix = os.path.splitext(os.path.join(control.stft_extraction_path, self.chb_metadata.name))[0]
        print("attempting to generate windows for ictal")
        generate_windows_for_class("ictal")
        print("attempting to generate windows for preictal")
        generate_windows_for_class("preictal")
        print("attempting to generate windows for interictal")
        generate_windows_for_class("interictal")

    def generate_windows(self, df, target_number, sampling_rate, write_path_prefix):
        start = 0
        end = len(df)
        window_size = (control.window_size * sampling_rate)

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
            stft_data = stft(raw.get_data(), window_size, verbose=True)
            if not stft_data.shape == (17, 3841, 2):
                print(f"bad shape for {os.path.join(write_path_prefix, f'{w_start}-{w_end}-stft.npy')}")
                continue
            input()
            output_path = os.path.join(write_path_prefix, f"{w_start}-{w_end}-stft.npy")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            np.save(output_path, stft_data)