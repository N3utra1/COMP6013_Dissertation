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
    extract_features = True

    global show_heat_plots # shows heat plots of the STFT for each file that contains a seizure
    show_heat_plots = False 

    global train_model # trains a single model using default params, must specifiy batch and epochs below
    train_model = False
    global epochs
    epochs = 4
    global batch_size
    batch_size = 32

    global tune_model # finds an optimal model through hyperparam tuning, must specify hyperparam_limits below
    tune_model = True 
    global hyperparam_limits 
    hyperparam_limits = {
        "model_parameters" : {
            "num_conv_layers" : [2, 6, 10],
            "num_dense_layers"  : [4, 8, 12],
            "dense_layer_size" : [64, 128, 256]
        },
        "training_parameters" : {
            "epoch" : [1],
            "batch_size" : [128]
        }
    }

    global load_model # loads a specific model
    # load_model = "./train_model.keras" 
    load_model = False

    global common_columns
    common_columns = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ"]

    global band_pass_low
    band_pass_low = [57, 63]

    global band_pass_high
    band_pass_high = [117, 123]

    global preictal_period # minuites
    preictal_period = 20

    global model
    model = "cnn"
       
    global extractor
    extractor = {
       "overwrite" : False,
       "threading" : False 
    }

    global target
    # chb24 is not a valid patient. target can either be a specific patient or all (True) patients
    # target = True 
    target = "chb06"

    global csv_path
    csv_path = os.path.abspath("/data/csv-chb-mit")  
    # csv_path = os.path.abspath("../csv-chb-mit-scalp-eeg-database")

    global stft_extraction_path
    # feature_extraction_path = "D:\csv-feature-chb-mit"
    stft_extraction_path = "/data/stft-chb-mit"

    global dataset_path
    dataset_path = os.path.abspath("/data/chb-mit-scalp-eeg-database-1.0.0")

    global model_save_path
    model_save_path = "/data/results"

    global warnings_path
    warnings_path = os.path.abspath("../warnings")
    # incriments number on end of file to avoid overwriting other warning files
    existing_files = glob(f"{warnings_path}*")
    if len(existing_files) > 0:
        warnings_path = f"{warnings_path}-{len(existing_files )}" 
