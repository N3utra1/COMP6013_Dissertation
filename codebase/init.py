import os
from load_dataset.CHB import CHB 
from extract_data.Extractor import Extractor


# load datasets into OOP Objects
chb_mit_path = os.path.abspath("../chb-mit-scalp-eeg-database-1.0.0")
loaded_chb = []
for dir in os.listdir(chb_mit_path):
    if os.path.isdir(os.path.join(chb_mit_path, dir)) and not (dir == "chb24"):
        chb = CHB(dir_name=dir, chb_mit_path=chb_mit_path, verbosity=True) 
        loaded_chb.append(chb)
        break #TEMP


# write data to disc as CSV files (feature extraction)
write_path = os.path.abspath("../csv-chb-mit-scalp-eeg-database")
for chb in loaded_chb:
    extractor = Extractor(chb=chb, write_path=write_path, overwrite=False)