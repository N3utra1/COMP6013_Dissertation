import mne
from pprint import pprint

path = r"C:\Users\willi\Documents\UNIVERSITY\COMP6013_Dissertation\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf"
raw = mne.io.read_raw_edf(path)

channel_1, times = raw[0, :]
print(channel_1)