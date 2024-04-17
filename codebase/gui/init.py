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


# helps with OOM errors:
try:
    physical_devices = tf.config.list_physical_devices('GPU') 
    if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

simulating = False
common_columns = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ"]
current_generator = None
current_data = pd.DataFrame()
guesses = 0
current_tick = 0
current_raw = None
time_vector = np.linspace(0, 15, 3841)
freq_vector = np.linspace(0, 128, 17)
stft_plot = None
eeg_file_prefix = os.path.normpath("/data/csv-chb-mit/chb06")
generators = None
model = None
smooth_timer = ui.timer(1, lambda: smooth_timer_tasks())



# opens the 3 periods as generators
class FileLinesGenerator:
    def __init__(self, filename):
        print(f"reading in new file: {filename}")
        print(f"this might take a moment...")
        self.index = 0
        with open(filename, 'r') as f:
            self.lines = f.readlines()
        del self.lines[0]
        print(f"file opened with {len(self.lines)} lines")
    def get_lines(self):
        offset = int(self.index + (1 * 256))
        return_value = self.lines[int(self.index): int(offset)] 
        self.index += int(1 * 256) 
        if self.index >= len(self.lines):
            self.index = 0  # reset index if end of file is reached
        return [line.split(",") for line in return_value] 
    def get_index(self): 
        return self.index


def change_current_class():
    global generators
    global current_generator 
    global current_class_toggle
    print(f"setting the current generator to {current_class_toggle.value}")
    current_generator = generators[current_class_toggle.value]
    refresh_current_data()


def refresh_current_data():
    global current_data
    current_data = pd.DataFrame()
    while len(current_data) < (30 * 256):
        pull_window()

def update_prediction(prediction):
    global predicted_class_toggle
    global guesses
    print("recieved prediction:")
    print("['interictal', 'preictal', 'ictal']")
    print(prediction)
    prediction = (np.argmax(prediction[0]) + 1)
    if prediction == current_class_toggle.value: 
        guesses += 1
    predicted_class_toggle.value = prediction

def refresh():
    global current_data
    global time_vector
    global freq_vector
    global stft_plot
    print("updating GUI")
    ui_container.clear()
    with ui_container: 
        with ui.row().classes("col-md-6"):
            with ui.pyplot(figsize=(8, 8)):
                try:
                    for channel in current_raw.get_data():
                        plt.plot(channel[::128], "-k", linewidth=0.5)
                    plt.title('EEG Data')
                    plt.xlabel('Time (Hz)')
                    plt.ylabel('Amplitude (mV)')
                    plt.xlim(0, 30)
                    plt.ylim(-600, 600)
                except Exception as e:
                    print("hit issue when drawing eeg plot")
                    raise e

            with ui.pyplot(figsize=(8, 8)):
                try:
                    plt.imshow(np.abs(np.mean(stft_plot, axis=0)), origin='lower', aspect='auto', cmap='hot', norm=LogNorm(), extent=[time_vector.min(), time_vector.max(), freq_vector.min(), freq_vector.max()])
                    plt.colorbar(label='Color scale')
                    plt.title('STFT plot')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Frequency (Hz)')
                    plt.ylim([0,128])
                except Exception as e:
                    print("hit issue when drawing stft plot")
                    print(e)

def pull_window():
    global current_data
    global current_generator
    global current_raw
    current_data = pd.concat([current_data, pd.DataFrame(current_generator.get_lines())], ignore_index=True)
    if len(current_data) > (30*256): 
        # trim off the start of current_data to give a moving affect
        current_data = current_data.iloc[len(current_data)-(30*256):,] 
    info = mne.create_info(ch_names=common_columns, sfreq=256)
    current_raw = mne.io.RawArray(current_data.transpose(), info)

def generate_stft_window():
    global stft_image
    global stft_plot
    global current_raw
    stft_plot = stft(current_raw.get_data(), 7680)
    stft_image = np.expand_dims(stft_plot, axis=0)


def smooth_timer_tasks():
    global window_slider
    global current_data
    global stft_plot
    global current_raw
    global current_tick
    global guesses

    # update slider
    try:
        window_slider.set_value((guesses / current_tick) * 100)
    except Exception as e:
        window_slider.set_value(0)

    # pull new windows 
    pull_window()

    # update stft data
    generate_stft_window()

    # update prediction
    try:
        print("prediction results")
        prediction_results = model.predict(stft_image)
        update_prediction(prediction_results)
    except Exception as e: 
        print("error when generating prediction")
        print("e")
        io.notification(e, timeout="30", multi_line=True)
    
    current_tick += 1
    # refresh page 
    refresh()



with ui.row().classes("w-full justify-around"):
    with ui.column():
        ui.label("TRUE CLASS:").classes("font-bold")
        current_class_toggle = ui.toggle({1: 'interictal', 2: 'preictal', 3: 'ictal'}, value=1, on_change=change_current_class)

    with ui.column():
        ui.label("PREDICTED CLASS:").classes("font-bold")
        predicted_class_toggle = ui.toggle({1: 'interictal', 2: 'preictal', 3: 'ictal'}).classes("disabled, pointer-events-none")

with ui.row().classes("w-1/3 self-center"):
    ui.label("ACCURACY").classes("font-bold")
    window_slider = ui.slider(min=0, max=100, value=0).classes("w-50").classes("pointer-events-none")

ui_container = ui.row().classes("w-full justify-around")

def init():
    global current_data
    global model
    global generators
    print("\n\ninit function running...")

    model_path = "/data/results-test/2.4.64/48.32.keras"
    print(f"loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("creating the generators")
    generators = {1 : FileLinesGenerator(os.path.join(eeg_file_prefix, "chb06_01", "interictal", "master.csv")), 
                  2 : FileLinesGenerator(os.path.join(eeg_file_prefix, "chb06_09", "preictal", "master.csv")),
                  3 : FileLinesGenerator(os.path.join(eeg_file_prefix, "chb06_01", "ictal", "master.csv"))}

    print("updating class and refreshing UI")
    change_current_class()

    print("populating data")
    while len(current_data) < (30 * 256):
        pull_window()
    generate_stft_window()

    refresh()


app.on_startup(lambda: init())
ui.run(favicon="🧠", title="Real-time Simulation")
