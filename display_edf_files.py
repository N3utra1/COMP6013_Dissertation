import os
import numpy as np
import mne
import matplotlib.pyplot as plt

#this was used to generate images of the EEG graphs for use in the presentation and writeup

file_path = "./chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_01.edf"
raw = mne.io.read_raw_edf(file_path)

raw.plot(n_channels=5, scalings='auto', title='EEG Data')

plt.show()