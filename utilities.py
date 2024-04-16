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
import ast
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    def __init__(self, path):
        self.path = path
        self.lines = self.load_lines()
        self.metrics = {
            "configuration" : self.get_configuration(),
            "training" : {
                "dataset_generation_start" : self.get_date(self.lines[13]),
                "dataset_generation_end" : self.get_date(self.lines[23]),
                "dataset_generation_duration" : self.get_date(self.lines[23]) - self.get_date(self.lines[22]),
                "model_compiling_start" :  self.get_date(self.lines[22]),
                "model_compiling_end" :  self.get_date(self.lines[23]),
                "model_compiling_duration" : self.get_date(self.lines[23]) - self.get_date(self.lines[22]),
                "model_saving_start" :  self.get_date(self.lines[28]),
                "model_saving_end" :  self.get_date(self.lines[29]),
                "model_saving_duration" : self.get_date(self.lines[29]) - self.get_date(self.lines[28]),
                "accuracy" : self.get_training_metric(self.lines[26]),
                "loss" : self.get_training_metric(self.lines[27])
            },
            "testing" : {
                "testing_start" : self.get_date(self.lines[30]),
                "testing_end" : self.get_date(self.lines[31]),
                "testing_duration" : self.get_date(self.lines[31]) - self.get_date(self.lines[30]),
                "confusion_matrix" : self.get_confusion_matrix(self.lines[33:36]),
                "normalized_confusion_matrix" : self.get_confusion_matrix(self.lines[37:40]),
                "precision" : self.get_report_values(1),
                "recall" : self.get_report_values(2),
                "f1" : self.get_report_values(3),
                "accuracy": self.get_testing_accuracy()
            }
        }
    
    def __repr__(self):
        return str(self.metrics)
    
    def get_date(self, x):
        return datetime.datetime.strptime(" ".join(x.split()[-2:]), "%Y-%m-%d %H:%M:%S.%f")

    def load_lines(self):
        lines = [] 
        with open(self.path, "r") as f:
            lines = f.readlines()
        return lines
    
    def get_configuration(self):
        def get_value(x): return x.split()[-1]
        a = self.lines[4:7]
        m = self.lines[8:11]
        return f"{get_value(a[0])}.{get_value(a[1])}.{get_value(a[2])}/{get_value(m[1])}.{get_value(m[0])}" # deliberate flipped index 

    def get_training_metric(self, x):
        a = ast.literal_eval(" ".join(x.split()[2:]))
        return sum(a) / len(a)
    
    def get_confusion_matrix(self, x):
        return ast.literal_eval(re.sub(r"\[\,\s", "[", re.sub(r"\s+", ", ", "".join(x).replace("\n", ""))))

    def get_report_values(self, c):
        l = [l.strip().split() for l in self.lines[41:51] if l != '\n']
        r = {
            "interictal" : l[1][c],
            "preictal" : l[2][c],
            "ictal" : l[3][c],
            "macro_average" : l[5][c+1],
            "weighted_average" : l[6][c+1]
        }
        return r
    
    def get_testing_accuracy(self):
        return self.lines[47].strip().split()[1]

def plot_resutls():

    def build_timings_table(reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\\begin{tabular}{llll}
Model & Compiling & Saving & Testing \\\\
"""
        suffix_table_template = """
\end{tabular}
\caption{Time duration metrics for each model in format \\%H:\\%M:\\%S.\\%f.}
\label{tab:my-table}
\end{table}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f"  {m['configuration']}    &      {m['training']['model_compiling_duration']}       &     {m['training']['model_saving_duration']}       &     {m['testing']['testing_duration']}        \\")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open("../recorded-metrics/timings-table", "w+") as f:
            f.writelines(s)
        print("timings table written")

    def build_training_metrics_table(reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\\begin{tabular}{lll}
Model & Training Accuracy & Training Loss \\\\
"""
        suffix_table_template = """
\end{tabular}
\caption{Accuracy and Loss metrics (average across epochs) during model training.}
\label{tab:training-metrics}
\end{table}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f"  {m['configuration']}    &      {m['training']['accuracy']}       &     {m['training']['loss']} \\\\")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open("../recorded-metrics/training-metrics-table", "w+") as f:
            f.writelines(s)
        print("training metrics table written")

    def build_testing_accuracy_table(reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\\begin{tabular}{ll}
Model & Testing Accuracy \\\\
"""
        suffix_table_template = """
\end{tabular}
\caption{Accuracy metrics obtained during testing.}
\label{tab:testing-accuracy}
\end{table}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f"  {m['configuration']}    &      {m['testing']['accuracy']}   \\\\")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open("../recorded-metrics/testing-accuracy-table", "w+") as f:
            f.writelines(s)
        print("testing accuracy table written")

    def build_cmatrix_table(reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\centering
\\begin{tabular}{llll}
Model                    & \multicolumn{3}{l}{Confusion Matrix} \\\\
"""
        suffix_table_template = """
\end{tabular}
\caption{Generated Confusion Matrix during the model tuning process.}
\label{tab:cmatrix}
\end{table}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f"""\multirow{{3}}{{*}}{{{m["configuration"]}}}   &  {m['testing']['confusion_matrix'][0][0]} & {m['testing']['confusion_matrix'][0][1]} & {m['testing']['confusion_matrix'][0][2]} \\\\
                                                                            &  {m['testing']['confusion_matrix'][1][0]} & {m['testing']['confusion_matrix'][1][1]} & {m['testing']['confusion_matrix'][1][2]} \\\\
                                                                            &  {m['testing']['confusion_matrix'][2][0]} & {m['testing']['confusion_matrix'][2][1]} & {m['testing']['confusion_matrix'][2][2]} \\\\ \\hline""")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open("../recorded-metrics/cmatrix-table", "w+") as f:
            f.writelines(s)
        print("cmatrix table written")

    def build_normed_cmatrix_table(reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\centering
\\begin{tabular}{llll}
Model                    & \multicolumn{3}{l}{Normalized Confusion Matrix} \\\\ 
"""
        suffix_table_template = """
\end{tabular}
\caption{Generated Normalized Confusion Matrix during the model tuning process.}
\label{tab:normed-cmatrix}
\end{table}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f"""\multirow{{3}}{{*}}{{{m["configuration"]}}}   &  {m['testing']['normalized_confusion_matrix'][0][0]} & {m['testing']['normalized_confusion_matrix'][0][1]} & {m['testing']['normalized_confusion_matrix'][0][2]} \\\\
                                                                            &  {m['testing']['normalized_confusion_matrix'][1][0]} & {m['testing']['normalized_confusion_matrix'][1][1]} & {m['testing']['normalized_confusion_matrix'][1][2]} \\\\
                                                                            &  {m['testing']['normalized_confusion_matrix'][2][0]} & {m['testing']['normalized_confusion_matrix'][2][1]} & {m['testing']['normalized_confusion_matrix'][2][2]} \\\\ \\hline""")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open("../recorded-metrics/normed-cmatrix-table", "w+") as f:
            f.writelines(s)
        print("normed cmatrix table written")

    def build_metric_table(metric, reports):
        prefix_table_template = """
\\begin{table}[H]
\centering
\centering
\\begin{tabular}{llllll}
Model & Interictal & Preictal & Ictal & Macro Average & Weighted Average \\\\
"""
        suffix_table_template = f"""
\end{{tabular}}
\caption{{{metric} metrics obtained during testing.}}
\label{{tab:normed-cmatrix}}
\end{{table}}"""
        middle = []
        for report in reports:
            m = report.metrics
            middle.append(f" {m['configuration']} & {m['testing'][metric]['interictal']} & {m['testing'][metric]['preictal']} & {m['testing'][metric]['ictal']} & {m['testing'][metric]['macro_average']} & {m['testing'][metric]['weighted_average']} \\\\")
        s = prefix_table_template + "\n".join(middle) + suffix_table_template
        with open(f"../recorded-metrics/{metric}-metric-table", "w+") as f:
            f.writelines(s)
        print(f"{metric} table written")





    def plot_timings(reports):
        configurations = [report.metrics["configuration"] for report in reports]
        compiling_duration = [report.metrics["training"]["model_compiling_duration"].total_seconds() for report in reports]
        saving_duration = [report.metrics["training"]["model_saving_duration"].total_seconds() for report in reports]
        testing_duration = [report.metrics["testing"]["testing_duration"].total_seconds() for report in reports]

        barWidth = 0.25
        
        br1 = np.arange(len(compiling_duration)) 
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2] 
        
        plt.barh(br1, compiling_duration, color ='r', height = barWidth, edgecolor ='grey', label ='Compiling') 
        plt.barh(br2, saving_duration, color ='g', height = barWidth, edgecolor ='grey', label ='Saving') 
        plt.barh(br3, testing_duration, color ='b', height = barWidth, edgecolor ='grey', label ='Testing') 
        
        plt.ylabel('Models', fontweight ='bold', fontsize = 15) 
        plt.yticks([r + barWidth for r in range(len(compiling_duration))], configurations)
        plt.xlabel('Time (Seconds)', fontweight ='bold', fontsize = 15) 
        
        plt.legend()
        plt.savefig("./figs/timings.png", bbox_inches="tight") 
        plt.close()

    def plot_training_metrics(reports):
        def create_accuracy_plots(reports):
            accuracy_dict = {}

            for obj in reports:
                architecture, model = obj.metrics["configuration"].split('/')
                accuracy = obj.metrics["training"]["accuracy"]

                x_label, y_label = model.split('.')

                if not architecture in accuracy_dict.keys(): accuracy_dict[architecture] = {}
                accuracy_dict[architecture][(x_label, y_label)] =  accuracy
            architectures = sorted(accuracy_dict.keys())
            x_labels = sorted(set(x for arch in architectures for x, y in accuracy_dict[arch].keys()))
            y_labels = sorted(set(y for arch in architectures for x, y in accuracy_dict[arch].keys()), reverse=True)
            heatmap_values = np.zeros((len(architectures), len(x_labels), len(y_labels)))

            for k, arch in enumerate(architectures):
                for i, x_label in enumerate(x_labels):
                    for j, y_label in enumerate(y_labels):
                        heatmap_values[k, i, j] = accuracy_dict[arch].get((x_label, y_label), 0)

            for k, arch in enumerate(architectures):
                plt.figure(figsize=(6, 5))
                plt.title(f"Heatmap for {arch}")
                plt.xlabel("Batch Size")
                plt.ylabel("Epochs")
                plt.xticks(range(len(x_labels)), x_labels)
                plt.yticks(range(len(y_labels)), y_labels)

                plt.imshow(heatmap_values[k], cmap='hot', interpolation='nearest', vmin=0.5, vmax=1.0)
                plt.colorbar(label="Accuracy")
                plt.savefig(f"./figs/heatmap_training_accuracy_{arch}.png")
                plt.close()
                print(f"./figs/heatmap_training_accuracy_{arch}.png saved")

        def create_loss_plots(reports):
            loss_dict = {}

            for obj in reports:
                architecture, model = obj.metrics["configuration"].split('/')
                loss = obj.metrics["training"]["loss"]

                x_label, y_label = model.split('.')

                if not architecture in loss_dict.keys(): loss_dict[architecture] = {}
                loss_dict[architecture][(x_label, y_label)] = loss 
            architectures = sorted(loss_dict.keys())
            x_labels = sorted(set(x for arch in architectures for x, y in loss_dict[arch].keys()))
            y_labels = sorted(set(y for arch in architectures for x, y in loss_dict[arch].keys()))
            heatmap_values = np.zeros((len(architectures), len(x_labels), len(y_labels)))

            for k, arch in enumerate(architectures):
                for i, x_label in enumerate(x_labels):
                    for j, y_label in enumerate(y_labels):
                        heatmap_values[k, i, j] = loss_dict[arch].get((x_label, y_label), 0)

            for k, arch in enumerate(architectures):
                plt.figure(figsize=(6, 5))
                plt.title(f"Heatmap for {arch}")
                plt.xlabel("Batch Size")
                plt.ylabel("Epochs")
                plt.xticks(range(len(x_labels)), x_labels)
                plt.yticks(range(len(y_labels)), y_labels)

                plt.imshow(heatmap_values[k], cmap='hot', interpolation='nearest', vmin=0.5, vmax=1.0)
                plt.colorbar(label="Loss")
                plt.savefig(f"./figs/heatmap_training_loss_{arch}.png")
                plt.close()
                print(f"./figs/heatmap_training_loss_{arch}.png saved")

        create_accuracy_plots(reports)
        create_loss_plots(reports)

    def plot_testing_accuracy(reports):
        accuracy_dict = {}

        for obj in reports:
            architecture, model = obj.metrics["configuration"].split('/')
            accuracy = obj.metrics["testing"]["accuracy"]

            x_label, y_label = model.split('.')

            if not architecture in accuracy_dict.keys(): accuracy_dict[architecture] = {}
            accuracy_dict[architecture][(x_label, y_label)] =  accuracy
        architectures = sorted(accuracy_dict.keys())
        x_labels = sorted(set(x for arch in architectures for x, y in accuracy_dict[arch].keys()))
        y_labels = sorted(set(y for arch in architectures for x, y in accuracy_dict[arch].keys()), reverse=True)
        heatmap_values = np.zeros((len(architectures), len(x_labels), len(y_labels)))

        for k, arch in enumerate(architectures):
            for i, x_label in enumerate(x_labels):
                for j, y_label in enumerate(y_labels):
                    heatmap_values[k, i, j] = accuracy_dict[arch].get((x_label, y_label), 0)

        # Create the heatmap using matplotlib
        for k, arch in enumerate(architectures):
            plt.figure(figsize=(6, 5))
            plt.title(f"Heatmap for {arch}")
            plt.xlabel("Batch Size")
            plt.ylabel("Epochs")
            plt.xticks(range(len(x_labels)), x_labels)
            plt.yticks(range(len(y_labels)), y_labels)

            plt.imshow(heatmap_values[k], cmap='hot', interpolation='nearest', vmin=0.5, vmax=1.0)
            plt.colorbar(label="Accuracy")
            plt.savefig(f"./figs/heatmap_testing_accuracy_{arch}.png")
            plt.close()
            print(f"./figs/heatmap_testing_accuracy_{arch}.png saved")

    def plot_metric(metric, reports):
        for target in ["interictal", "preictal", "ictal", "macro_average", "weighted_average"]:
            d = {}

            for obj in reports:
                architecture, model = obj.metrics["configuration"].split('/')
                value = obj.metrics["testing"][metric][target]

                x_label, y_label = model.split('.')

                if not architecture in d.keys(): d[architecture] = {}
                d[architecture][(x_label, y_label)] = value 
            architectures = sorted(d.keys())
            x_labels = sorted(set(x for arch in architectures for x, y in d[arch].keys()))
            y_labels = sorted(set(y for arch in architectures for x, y in d[arch].keys()), reverse=True)
            heatmap_values = np.zeros((len(architectures), len(x_labels), len(y_labels)))

            for k, arch in enumerate(architectures):
                for i, x_label in enumerate(x_labels):
                    for j, y_label in enumerate(y_labels):
                        heatmap_values[k, i, j] = d[arch].get((x_label, y_label), 0)

            # Create the heatmap using matplotlib
            for k, arch in enumerate(architectures):
                plt.figure(figsize=(6, 5))
                plt.title(f"Heatmap for {arch}")
                plt.xlabel("Batch Size")
                plt.ylabel("Epochs")
                plt.xticks(range(len(x_labels)), x_labels)
                plt.yticks(range(len(y_labels)), y_labels)

                plt.imshow(heatmap_values[k], cmap='hot', interpolation='nearest', vmin=0.5, vmax=1.0)
                plt.colorbar(label=f"{target}")
                plt.savefig(f"./figs/heatmap_{metric}_{target}_{arch}.png")
                plt.close()
                print(f"./figs/heatmap_{metric}_{target}_{arch}.png saved")



    reports = glob.glob("/data/results-test/*/*.log")
    good_reports = []
    for report in reports:
        lines = []
        with open(report, "r") as f:
            lines = f.readlines()
        try:
            l = lines[35]
        except:
            print(f"removed {report} (line count)")
            continue 

        if ast.literal_eval(re.sub(r"\[\,\s", "[", re.sub(r"\s+", ", ", "".join(lines[33:36]).replace("\n", ""))))[-1][-1] == 0:
            print(f"removed {report} (bad cmatrix)")
            continue


        if re.search("\[.*\]", l):
            o = DataExtractor(report)
            good_reports.append(o)
    
    # build_timings_table(good_reports)
    # build_training_metrics_table(good_reports)
    # build_testing_accuracy_table(good_reports)
    # build_cmatrix_table(good_reports)
    # build_normed_cmatrix_table(good_reports)
    # build_metric_table("precision", good_reports)
    # build_metric_table("recall", good_reports)
    # build_metric_table("f1", good_reports)


    # plot_timings(good_reports)
    # plot_training_metrics(good_reports)
    # plot_testing_accuracy(good_reports)
    plot_metric("precision", good_reports)
    plot_metric("recall", good_reports)
    plot_metric("f1", good_reports)



control.init()
# method_menu()
plot_resutls()