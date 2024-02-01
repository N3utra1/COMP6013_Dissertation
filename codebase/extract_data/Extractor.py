import mne
from mne.time_frequency import stft, stftfreq
import os
import pandas as pd
import threading
import control
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
        def show_plot(chb_file, stft_data, max_time):
            time_vector = np.linspace(0, max_time, stft_data.shape[1])
            freq_vector = np.linspace(0, chb_file.data_sampling_rate/2, stft_data.shape[0])

            plt.imshow(np.abs(np.mean(stft_data, axis=0)), origin='lower', aspect='auto', cmap='hot', norm=LogNorm(), extent=[time_vector.min(), time_vector.max(), freq_vector.min(), freq_vector.max()])
            plt.colorbar(label='Color scale')
            plt.title(f'{chb_file.name} STFT results')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim([0,128])
            plt.show()


        for chb_file in self.chb_metadata.edf_files:
            # get corresponding csv path
            csv_suffix = os.path.join(*os.path.normpath(chb_file.path).split(os.sep)[-2:]).replace("edf", "csv")
            chb_csv_path = os.path.join(self.csv_path, csv_suffix)
            chb_feature_write_path = os.path.join(self.write_path, csv_suffix)
            eeg_data_df = pd.read_csv(chb_csv_path)
            info = mne.create_info(ch_names=control.common_columns, sfreq=chb_file.data_sampling_rate)
            raw = mne.io.RawArray(eeg_data_df[control.common_columns].transpose(), info)

            wsize = chb_file.data_sampling_rate * control.window_size
            stft_data = stft(raw.get_data(), wsize)
            if not chb_file.seizure_dict == {} and control.show_heat_plots:
                max_time = len(eeg_data_df) / chb_file.data_sampling_rate 
                print(chb_file)
                show_plot(chb_file, stft_data, max_time)