import os
from load_dataset.CHB import CHB 

chb_mit_path = "../chb-mit-scalp-eeg-database-1.0.0"
loaded_chb = []
for dir in os.listdir(chb_mit_path):
    if os.path.isdir(os.path.join(chb_mit_path, dir)) and not (dir == "chb24"):
        chb = CHB(dir_name=dir, chb_mit_path=os.path.abspath(chb_mit_path), verbosity=True) 
        loaded_chb.append(chb)