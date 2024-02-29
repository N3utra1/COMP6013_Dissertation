import os
from glob import glob

def warning(message):
    with open(warnings_path, "a") as file:
        print(message)
        file.write(f"{message}\n")

def init():
    global seed
    seed = 111

    # the STFT window size in seconds
    global window_size
    window_size = 30

    global test_percentage
    test_percentage = 0.2

    global load_dataset # this needs to be true to be able to write and extract the data
    load_dataset = False 
    global write_dataset # this writes the 3 ictal periods to csv file, containing the common columns (channels) with the band pass filter applied
    write_dataset = False 
    global extract_features # this balances the 3 ictal periods and generates spectograph images for each channel in 30 second segments
    extract_features = False 
    global show_heat_plots # shows heat plots of the STFT for each file that contains a seizure
    show_heat_plots = False 
    global train_model 
    train_model = True

    global common_columns
    common_columns = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ"]

    global band_pass_low
    band_pass_low = [57, 63]

    global band_pass_high
    band_pass_high = [117, 123]

    global preictal_period # minuites
    preictal_period = 20

    global avaliable_models
    avaliable_models = [
       "cnn"
    ]
       
    global model 
    model = "cnn"

    global batch_size
    batch_size = 421  # should be 1, 19, 421 as total files are 7999 for each class

    global epoch 
    epoch = 1

    global extractor
    extractor = {
       "overwrite" : False,
       "threading" : False 
    }

    global target
    # chb24 is not a valid patient. target can either be a specific patient or all (True) patients
    #target = True 
    target = "chb06"

    global csv_path
    csv_path = os.path.abspath("D:\csv-chb-mit")  
    # csv_path = os.path.abspath("../csv-chb-mit-scalp-eeg-database")

    global stft_extraction_path
    # feature_extraction_path = "D:\csv-feature-chb-mit"
    stft_extraction_path = "D:\stft-chb-mit"

    global dataset_path
    dataset_path = os.path.abspath("../chb-mit-scalp-eeg-database-1.0.0")

    global warnings_path
    warnings_path = os.path.abspath("../warnings")
    # incriments number on end of file to avoid overwriting other warning files
    existing_files = glob(f"{warnings_path}*")
    if len(existing_files) > 0:
        warnings_path = f"{warnings_path}-{len(existing_files )}" 
