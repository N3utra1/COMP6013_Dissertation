#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from nicegui import Client, app, core, run, ui
from keras.models import load_model
import os
from mne.time_frequency import stft
import mne
import tensorflow as tf


# simulating controls the model
simulating = False
common_columns = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ"]
current_generator = None

# opens the 3 periods as generators
class FileLinesGenerator:
    def __init__(self, filename):
        print(f"reading in new file: {filename}")
        print(f"this might take a moment...")
        self.index = 0
        with open(filename, 'r') as f:
            self.lines = f.readlines()
        del self.lines[0]
    def get_lines(self):
        try:
            return_value = self.lines[self.index: self.index + 30*256]
            self.index += len(return_value)
            return [line.split(",") for line in return_value] 
        except StopIteration:
            self.index = 0
            self.get_lines()
    def get_index(self): 
        return self.index
eeg_file_prefix = os.path.normpath("D:\csv-chb-mit\chb06\chb06_01")
generators = {1 : FileLinesGenerator(os.path.join(eeg_file_prefix, "interictal", "master.csv")), 
              2 : FileLinesGenerator(os.path.join(eeg_file_prefix, "preictal", "master.csv")),
              3 : FileLinesGenerator(os.path.join(eeg_file_prefix, "ictal", "master.csv"))}
def change_current_class():
    global current_generator 
    global current_class_toggle
    current_generator = generators[current_class_toggle.value]
    update()


# set up the timer which runs on an speed of 30 seconds 
# timer can be turned on and off with the "toggle" button
timer = ui.timer(30, lambda: update())

with ui.row().classes("w-full justify-around"):
    with ui.column():
        ui.label("TOGGLE SIMULATION:").classes("font-bold")
        ui.switch().bind_value_to(timer, 'active')

    with ui.column():
        ui.label("TRUE CLASS:").classes("font-bold")
        current_class_toggle = ui.toggle({1: 'interictal', 2: 'preictal', 3: 'ictal'}, value=1, on_change=change_current_class)

    with ui.column():
        ui.label("PREDICTED CLASS:").classes("font-bold")
        predicted_class_toggle = ui.toggle({1: 'interictal', 2: 'preictal', 3: 'ictal'}).classes("disabled, pointer-events-none")
def update_prediction(prediction):
    global predicted_class_toggle
    print("recieved prediction:")
    print(prediction)
    predicted_class_toggle.value = 1 


model_path = "D:\\results\\2.4.128\\16.8.keras"
print(f"opening {model_path}")
print("exists? : ")
print(os.path.exists(model_path))
model = tf.keras.models.load_model(model_path)


ui_container = ui.row().classes("w-full justify-around")

def update():
    global current_class_toggle
    global current_generator
    global common_columns
    global model
    print("updating the GUI")

    current_data = pd.DataFrame(current_generator.get_lines())
    print(current_data)
    info = mne.create_info(ch_names=common_columns, sfreq=256)
    raw = mne.io.RawArray(current_data.transpose(), info, verbose=True)

    stft_plot= stft(raw.get_data(), 7680)
    print(stft_plot.shape)

    ui_container.clear()
    with ui_container: 
        with ui.row().classes("col-md-6"):
            with ui.pyplot(figsize=(8, 8)):
                for channel in raw.get_data():
                    downsampled_data = channel[::30]
                    plt.plot(downsampled_data, "-k", linewidth=0.1)
                plt.title('EEG Data')
                plt.xlabel('Time (Hz)')
                plt.ylabel('Amplitude (mV)')
                plt.xlim(0, 30)
                plt.ylim(-600, 600)

            with ui.pyplot(figsize=(8, 8)):
                time_vector = np.linspace(0, 15, 3841)
                freq_vector = np.linspace(0, 128, 17)

                plt.imshow(np.abs(np.mean(stft_plot, axis=0)), origin='lower', aspect='auto', cmap='hot', norm=LogNorm(), extent=[time_vector.min(), time_vector.max(), freq_vector.min(), freq_vector.max()])
                plt.colorbar(label='Color scale')
                plt.title('STFT plot')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.ylim([0,128])

    update_prediction(model.predict(stft_plot))

change_current_class()
update()
ui.run()