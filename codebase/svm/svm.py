# import mne
# from pprint import pprint

# path = r"C:\Users\willi\Documents\UNIVERSITY\COMP6013_Dissertation\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf"
# raw = mne.io.read_raw_edf(path)

# print(raw.info)
# target_channel = "P7-O1"
# target_second = 10

# # Get the data for the specified channel and time range
# channel_data, times = raw.pick_channels([target_channel]).crop(tmin=target_second, tmax=target_second+ 1.0)[:]

# # Print the data
# print(f"Data for {target_channel} from {target_second} to {target_second+ 1.0} seconds:")
# print(channel_data)



# import mne
# import numpy as np
# import pywt 
# from scipy.stats import skew, kurtosis
# from scipy.signal import welch
# from pyentrp import entropy
# from mne.time_frequency import psd_welch
# from mne.features import compute_feature_dict

# # Load EEG data
# path = 'path/to/your/file.edf'
# raw = mne.io.read_raw_edf(path, preload=True)

# # Define a time window for feature extraction
# start_time = 10.0
# end_time = 20.0
# raw_crop = raw.crop(tmin=start_time, tmax=end_time)

# # Extract features
# features = {}

# # Frequency domain features
# psd, freqs = psd_welch(raw_crop)
# features['psd_alpha'] = np.mean(psd[:, (freqs >= 8) & (freqs <= 13)])
# features['psd_beta'] = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])

# # Time domain features
# data, _ = raw_crop[:, :]
# features['mav'] = np.mean(np.abs(data))
# features['rms'] = np.sqrt(np.mean(data**2))
# features['std'] = np.std(data)

# # Statistical features
# features['skewness'] = skew(data)
# features['kurtosis'] = kurtosis(data)

# # Entropy measures
# features['shannon_entropy'] = entropy.shannon_entropy(data)
# features['approximate_entropy'] = entropy.approximate_entropy(data, 2, 0.2)

# # Wavelet Transform Coefficients
# wavelet = 'db1'  # Choose a wavelet family and type
# coeffs, _ = pywt.cwt(data, np.arange(1, 128), wavelet)
# features['wavelet_coefficients'] = np.mean(coeffs, axis=1)  # You might choose a different aggregation method

# # Print extracted features
# for feature_name, value in features.items():
#     print(f"{feature_name}: {value}")

import mne
import numpy as np
import pandas as pd

# Load EEG data
path = r"C:\Users\willi\Documents\UNIVERSITY\COMP6013_Dissertation\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf"
raw = mne.io.read_raw_edf(path, preload=True)

all_channels = raw.ch_names

# Create an empty DataFrame to store the data for all channels
df_all_channels = pd.DataFrame()

# Loop through all channels and extract data
print(raw.ch_names)
for channel_name in all_channels:
    print(channel_name)
    selected_channel = raw.pick(channel_name)
    data, times = selected_channel[:, :]
    df_channel = pd.DataFrame(data.T, columns=[channel_name])
    df_all_channels = pd.concat([df_all_channels, df_channel], axis=1)

csv_path = r"./output.csv"
df_all_channels.to_csv(csv_path, index=False)

# raw_new_ref = mne.add_reference_channels(raw, ref_channels=["EEG 999"])
# # plots raw EEG signals
# raw_new_ref.plot()
# input()


# # plots the specral density 
# spectrum = raw.compute_psd()
# spectrum.plot(average=True, picks="data", exclude="bads")
# input()
