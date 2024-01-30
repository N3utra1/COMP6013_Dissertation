import mne
import os
import pandas as pd
import threading
import control
import numpy as np

from scipy.signal import stft
import matplotlib.pyplot as plt


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
            # get corresponding csv path
            csv_suffix = os.path.join(*os.path.normpath(chb_file.path).split(os.sep)[-2:]).replace("edf", "csv")
            chb_csv_path = os.path.join(self.csv_path, csv_suffix)
            chb_feature_write_path = os.path.join(self.write_path, csv_suffix)
            eeg_data_df = pd.read_csv(chb_csv_path)
            # Zxx is the STFT of eeg_data_df
            sample_frequencies, segment_times, Zxx= stft(eeg_data_df, fs=chb_file.data_sampling_rate, nperseg=control.window_size * chb_file.data_sampling_rate) 
            stft_channel_df = pd.DataFrame(np.abs(Zxx), columns=segment_times, index=sample_frequencies)
            stft_channel_df = stft_channel_df.T.reset_index()
            stft_channel_df = stft_channel_df.rename(columns={'index': 'Time'})
            stft_channel_df.to_csv(chb_feature_write_path)
            plt.pcolormesh(stft_channel_df.index, stft_channel_df.columns, np.log1p(stft_channel_df.T))
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.colorbar(label='Log Magnitude')
            plt.title(f"STFT of EEG Data for {csv_suffix}")
            file_path = chb_feature_write_path.replace("csv", "png")
            plt.savefig(file_path)
            plt.close()
