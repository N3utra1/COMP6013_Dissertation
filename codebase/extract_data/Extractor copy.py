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
        def count_lines(filename):
            try:
                with open(filename, 'r') as f:
                    for i, line in enumerate(f, 1):
                        pass
                return i - 1
            except:
                return 0 

        def process_class(class_type):        
            line_count = 0
            for chb_file in self.chb_metadata.edf_files:
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                count = count_lines(os.path.join(csv_prefix, class_type, "master.csv"))
                line_count += count 
                print(f"{count} for {chb_file.name} for class {class_type}")

            print(f"{class_type}: {(line_count / frequency) / 60} mins")
            if window_size > line_count:
                print(f"not enough {class_type} data for subject {chb_file.name}. Throwing error")
                raise RuntimeError
            print("total {line_count} for class {class_type}")
            return line_count

        frequency = self.chb_metadata.edf_files[0].data_sampling_rate 
        window_size = control.window_size * frequency
        ictal_count = process_class("ictal")
        print()
        preictal_count = process_class("preictal")
        print()
        interictal_count = process_class("interictal")

        dataframes = {'ictal': ictal_count, 'preictal': preictal_count, 'interictal': interictal_count}
        largest_df_name = max(dataframes, key=lambda x: len(dataframes[x]))
        largest = dataframes[largest_df_name]
        target_number_of_windows = len(largest) // window_size

        stft_prefix = os.path.splitext(os.path.join(control.stft_extraction_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]

        ictal_all = pd.DataFrame()
        preictal_all = pd.DataFrame()
        interictal_all = pd.DataFrame()
        for chb_file in self.chb_metadata.edf_files:
            try:
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                current_ictal = pd.read_csv(os.path.join(csv_prefix, "ictal", "master.csv"), skiprows=1)
                ictal_all.append(current_ictal)
            except:
                pass

            try:
                current_preictal = pd.read_csv(os.path.join(csv_prefix, "preictal", "master.csv"), skiprows=1)
                preictal_all.append(current_preictal)
            except:
                pass

            try:
                current_interictal = pd.read_csv(os.path.join(csv_prefix, "interictal", "master.csv"), skiprows=1)
                interictal_all.append(current_interictal)
            except:
                pass

        print(f"starting ictal")
        self.generate_windows(ictal_all, target_number_of_windows, frequency, os.path.join(stft_prefix, "ictal"))

        print(f"starting preictal")
        self.generate_windows(preictal_all, target_number_of_windows, frequency, os.path.join(stft_prefix, "preictal"))

        print(f"starting interictal")
        self.generate_windows(interictal_all, target_number_of_windows, frequency, os.path.join(stft_prefix, "interictal"))


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
            stft_data = stft(raw.get_data(), window_size, verbose=False)
            if not stft_data.shape == (17, 3841, 2):
                print(f"bad shape for {os.path.join(write_path_prefix, f'{w_start}-{w_end}-stft.npy')}")
                input()
            output_path = os.path.join(write_path_prefix, f"{w_start}-{w_end}-stft.npy")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            np.save(output_path, stft_data)