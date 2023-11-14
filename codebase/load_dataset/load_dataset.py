from pyedflib import highlevel
import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def plot_eeg(signals, signal_headers, header):
    n = len(signals)
    fig = plt.figure(figsize=(150, 50))
    ax = plt.axes()
    for i in np.arange(n):
        ax.plot(signals[i], color='purple')
    plt.show()


# Directory containing the EDF files
directory = "C:/Users/willi/Documents/UNIVERSITY/COMP6013_Dissertation/chb-mit-scalp-eeg-database-1.0.0/chb02/"


loaded_data = {}
# for each file in dir
for filename in os.listdir(directory):
    if filename.endswith(".edf"):
        print(f"loading {filename}")
        path = os.path.join(directory, filename)

        # load the data 
        signals, signal_headers, header = highlevel.read_edf(path)
        loaded_data.update({filename : {"signals": signals, "signal_headers": signal_headers, "header" : header}})

        input()
    else:
        print(f"skipping {filename} as not EDF")


