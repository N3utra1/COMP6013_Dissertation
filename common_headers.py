import os
import pandas as pd
import copy
from codebase import control

#        This is a simple utility program that was used to extract 
#        EEG headers which are contained in all of the files.

control.init()

original = []
headers = []
removed = []
first_done = False
for chb_dir in os.listdir(control.csv_path):
    sub_path = os.path.join(control.csv_path, chb_dir)
    for chb in os.listdir(sub_path):
        if chb.endswith(".csv"):
            print(f"reading {chb}")
            file_path = os.path.join(sub_path, chb)
            df = pd.read_csv(file_path, nrows=1)

            if not first_done: 
                headers = df.columns.tolist()
                original = copy.deepcopy(headers)
                first_done = True
                continue 
            current_files_headers = df.columns.tolist()
            new_headers = copy.deepcopy(headers)
            for name in headers:
                if not name in current_files_headers:
                    new_headers.remove(name)
                    removed.append(name)
                    print(f"removed {name}")
            headers = new_headers
print(f"original ({len(original)}):\n{original}")
print(f"remaining ({len(headers)}):\n{headers}")
print(f"removed ({len(removed)}):\n{removed}")