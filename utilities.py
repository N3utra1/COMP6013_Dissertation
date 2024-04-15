"""
    This is used to debug files written by the program. It is not needed when running the finished product
"""
from pprint import pprint
import mne
from mne.time_frequency import stft
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np 
import os
import glob
from codebase import control
import subprocess

files_to_check = glob.glob("D:\\stft-chb-mit\\chb06\\*\\*.npy")



def check_shape():
    # checks if there is a file saved which is not the appropriate shape for the ml model
    for file in files_to_check:
        current_file = np.load(file)
        if not current_file.shape == (17, 3841, 2):
            print(f"bad shape for {file}")
            print(current_file.shape)
        del(current_file)

def check_3rd_dim():
    # check to see if the 3rd dim has data in it (debugging)
    for file in files_to_check:
        current_file = np.load(file)
        if current_file.shape[2] == 0:
            print(f"{file}")
        del(current_file)

def check_2nd_dim_chb06():
    files_to_check = glob.glob("D:\\stft-chb-mit\\chb06\\*\\*.npy")
    for file in files_to_check:
        current_file = np.load(file)
        for x in range(current_file.shape[0]):
            for y in range(current_file.shape[1]):
                print(current_file[x][y])
        del(current_file)

def open_windows():
    def menu():
        def get_path(path, level):
            if level == 3:
                return path
            indent = "\t" * level
            print(f"\n{indent}path: {path}")
            results = glob.glob(path + "*")
            if not results:
                return path
            
            page_size = 10
            num_pages = (len(results) + page_size - 1) // page_size
            page = 1

            while True:
                print(f"Page {page}/{num_pages}:")
                start_index = (page - 1) * page_size
                end_index = min(page * page_size, len(results))
                for i in range(start_index, end_index):
                    item = results[i].split(os.path.sep)[-1]
                    print(f"{indent}{i + 1}. {item}")

                choice = input("Enter the number to select, 'p' for previous page, 'n' for next page, or 'q' to quit: ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(results):
                        new_path = results[choice - 1]
                        if not os.path.isdir(new_path):
                            show_plot(new_path)
                        path = get_path(new_path + os.path.sep, level + 1)
                elif choice.lower() == 'p':
                    if page > 1:
                        page -= 1
                elif choice.lower() == 'n':
                    if page < num_pages:
                        page += 1
                elif choice.lower() == 'q':
                    method_menu() 

        path = get_path("D:\\stft-chb-mit\\", 0)
        if path:
            print(f"\nSelected path: {path}")
            show_plot(path)
        else:
            print("No file selected.")

    def show_plot(file):
        stft_data = np.load(file)

        time_vector = np.linspace(0, 15, 3841)
        freq_vector = np.linspace(0, 128, 17)

        plt.imshow(np.abs(np.mean(stft_data, axis=0)), origin='lower', aspect='auto', cmap='hot', norm=LogNorm(), extent=[time_vector.min(), time_vector.max(), freq_vector.min(), freq_vector.max()])
        plt.colorbar(label='Color scale')
        plt.title(f'{file} STFT results')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim([0,128])
        plt.show()

    menu()


def method_menu():
    print("Available methods:")
    print("1. check_shape()")
    print("2. check_3rd_dim()")
    print("3. check_2nd_dim_chb06()")
    print("4. open_windows()")
    method_choice = input("Enter the number to select the method, or 'q' to go back to the previous menu: ").strip()
    if method_choice == '1':
        check_shape()
    elif method_choice == '2':
        check_3rd_dim()
    elif method_choice == '3':
        check_2nd_dim_chb06()
    elif method_choice == '4':
        open_windows()
    elif method_choice.lower() == 'q':
        return None
    else:
        print("Invalid choice. Please try again.")
        method_menu()


def calculate_class_size():
    classes = ["interictal", "ictal", "preictal"]
    raw_recording_path = "/data/csv-chb-mit/"
    subjects = [path.split(os.sep)[-1] for path in glob.glob(os.path.join(raw_recording_path, "*"))]
    d = dict.fromkeys(subjects)
    for subject in d.keys():
        d[subject] = dict.fromkeys(classes)


    for subject in subjects:
        print(f"calculating values for {subject}")
        for c in classes:
            paths = glob.glob(os.path.join(raw_recording_path, subject, "*", c, "master.csv"))
            total_size = 0
            i = 0
            for path in paths:
                i += 1
                total_size += int(subprocess.run(["wc", "-l", path], capture_output=True, text=True).stdout.split()[0])
                print(f"{c} file {i} / {len(paths)}", end="\r")
            print(c) 
            d[subject][c] = {
                "num_files" : len(paths),
                "line_count" : total_size,
                "num_seconds" : (total_size / 256),
                "num_mins" : ((total_size / 256 ) / 60),
            }
        
    pprint(d)


import matplotlib.pyplot as plt
import re
import datetime

class DataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.extract_data()

    def extract_data(self):
        data = {}
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    data[key.strip()] = value.strip()
                elif 'accuracy' in line:
                    data['training accuracy'] = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                elif 'loss' in line:
                    data['training loss'] = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
        return data

    def plot_metrics(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Training Metrics')
        axs[0].plot(self.data['training accuracy'], label='accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()
        axs[1].plot(self.data['training loss'], label='loss', color='orange')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend()
        plt.show()

# data_extractor = DataExtractor('path_to_your_file.txt')
# data_extractor.plot_metrics()


def plot_resutls():
    reports = glob.glob("/data/results-test/*/*.log")
    good_reports = []
    for report in reports:
        print(report)
        lines = []
        with open(report, "r") as f:
            lines = f.readlines()
        try:
            l = lines[35]
        except:
            continue 
            
        if re.search("\[.*\]", l):
            good_reports.append(lines)

    for report in good_reports:
        print(len(report))



control.init()
# method_menu()
plot_resutls()