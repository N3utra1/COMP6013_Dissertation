import os
from glob import glob
import pandas as pd
import control
from operator import itemgetter

class Feature_Writer:
    def __init__(self, epoch=256):
        self.epoch = epoch
        pass
    
    def process_lines(self, lines):
        lines = [float(value) for value in lines]
        features = []

        # remove hz 47-53, hz 97-103 and hz 0



        return lines + features

def main():
    # setup feature writer
    epoch = (256 * 30)
    feature_writer = Feature_Writer(epoch=epoch)


    # open each file and append features
    # csv_path = control.csv_path
    csv_path = r"C:\Users\willi\Documents\UNIVERSITY\COMP6013_Dissertation\csv-chb-mit-scalp-eeg-database"

    file_paths = glob(os.path.join(csv_path, "**/*.csv"))
    for file in file_paths:
        file_content = []
        current_file = open(file, "r")
        line_number = 0
        current_epoch = []

        header_line = current_file.readline().split(",")
        header_line[-1] = header_line[-1].strip()
        selected_indexs = [header_line.index(name) for name in control.common_columns]

        while True:
            line_number += 1
            print(line_number)
            current_line = current_file.readline()
            if not current_line: break
            current_line = current_line.split(",")
            current_line = list(itemgetter(*selected_indexs)(current_line))
            current_epoch.append(current_line)
            if not line_number % epoch:
                file_content += feature_writer.process_lines(current_epoch)
                current_epoch = []
                for line in file_content[-epoch:]:
                    print(line)
                input()


if __name__ == "__main__":
    control.init()
    main()