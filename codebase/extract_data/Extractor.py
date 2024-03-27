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
        if self.chb_metadata.name == "chb06": return
        if not self.quick_ictal_length_check(): return
        print(f"overwrite is set to {self.overwrite}")
        if threading:
            try:
                self.extract_threaded() == -1
            except ValueError:
                print(f"Error when generating STFT for {self.chb_metadata.name}") 
        else: 
            try:
                self.extract()
            except ValueError:
                print(f"Error when generating STFT for {self.chb_metadata.name}") 

    def quick_ictal_length_check(self):
        # if there is not enough ictal data for 10 windows skip subject
        data_sampling_rate = self.chb_metadata.edf_files[0].data_sampling_rate
        seizures = self.chb_metadata.seizures
        total_length = 0
        for seizure in seizures.values():
            for s in seizure.values():
                total_length += (s["end"] - s["start"]) * data_sampling_rate 
        if total_length < 10 * (30 * data_sampling_rate):
            print(f"skipping {self.chb_metadata.name} subject due to insufficent ictal data ({total_length // data_sampling_rate}s, target is 300s)")
            return False 
        return True

    def extract_threaded(self):
        print("threading is not implimented, falling back to single threaded extraction")
        return self.extract()

    def extract(self):
        def count_lines(filename):
            with open(filename, 'r') as f:
                for i, line in enumerate(f, 1):
                    pass
            return i - 1

        def get_class_size(class_type):
            total = 0
            i = 1
            for chb_file in self.chb_metadata.edf_files:
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                file_path = os.path.join(csv_prefix, class_type, "master.csv")
                if os.path.exists(file_path):
                    total += count_lines(file_path)
                print(f"total time samples: {total} ({i}/{len(self.chb_metadata.edf_files)})", end="\r")
                i += 1
            print()
            return total

        def generate_windows_for_class(class_type):
            all_dfs = []
            i = 1
            for chb_file in self.chb_metadata.edf_files:
                print(f"reading {i}/{len(self.chb_metadata.edf_files)}", end="\r")
                csv_prefix = os.path.splitext(os.path.join(control.csv_path, *os.path.normpath(chb_file.path).split(os.sep)[-2:]))[0]
                path = os.path.join(csv_prefix, class_type, "master.csv")
                if os.path.exists(path): all_dfs.append(pd.read_csv(path))
                i += 1
            print()
            whole_df = pd.concat(all_dfs) 
            return -1 if len(whole_df) < ((control.window_size * 256) // 2) else self.generate_windows(whole_df, target_number_of_windows, 256, os.path.join(stft_prefix, class_type))

        dataframes = {'ictal': 1, 'preictal': 1, 'interictal': get_class_size("interictal")} # this forces it to always think interictal is the largest class
        print(f"target number of windows: {(dataframes['interictal'] // 7680) // 30} (total ticks {dataframes['interictal']})")
        largest_df_name = max(dataframes, key=lambda x: dataframes[x])
        largest = dataframes[largest_df_name]
        target_number_of_windows = largest // (control.window_size * 256)
        stft_prefix = os.path.splitext(os.path.join(control.stft_extraction_path, self.chb_metadata.name))[0]
        try:
            generate_windows_for_class("ictal")
            generate_windows_for_class("preictal")
            generate_windows_for_class("interictal")
        except ValueError as e:
            raise e
            

    def generate_windows(self, df, target_number, sampling_rate, write_path_prefix):
        start = 0
        end = len(df)
        window_size = (control.window_size * sampling_rate) # 30 second window should be 7680

        file, mode = os.path.split(write_path_prefix)[-2:]
        print(f"writing {mode} for {file}")
        step = (end - start - window_size) / (target_number- 1)
        windows = [(int(i * step), int(i * step + window_size)) for i in range(target_number)]

        i = 1
        for window in windows:
            w_start = window[0]
            w_end = window[1]
            output_path = os.path.join(write_path_prefix, f"{w_start}-{w_end}-stft.npy")
            print(f"[{self.chb_metadata.name}/{mode}] creating stft image {i}/{len(windows)}", end="\r")
            i += 1
            if (not self.overwrite) and os.path.exists(output_path): continue

            df_window_slice = df.iloc[w_start : w_end]
            info = mne.create_info(ch_names=control.common_columns, sfreq=sampling_rate)
            raw = mne.io.RawArray(df_window_slice[control.common_columns].transpose(), info, verbose=False)

            stft_data = stft(raw.get_data(), window_size)

            if stft_data.shape == (17, 3841, 2):
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                if self.overwrite or not os.path.exists(output_path):
                    np.save(output_path, stft_data)
            else:
                print(f"shape of input data: {raw.get_data().shape}")
                print(f"bad shape for {output_path}, therefore skipping")
                print(f"output shape of {stft_data.shape}")
                control.warning(f"bad stft window shape for {output_path} : {stft_data.shape}")
        print()