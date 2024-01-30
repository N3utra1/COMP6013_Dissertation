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

    global load_dataset
    load_dataset = False 
    global write_dataset 
    write_dataset = False 
    global extract_features
    extract_features = True 
    write_dataset = False 
    global train_model 
    train_model = False 

    global common_columns
    common_columns = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ"]

    global avaliable_models
    avaliable_models = [
       "ann"
    ]
       
    global model 
    model = "ann"

    global incrimental_learning_batch_size
    incrimental_learning_batch_size = True

    global extractor
    extractor = {
       "overwrite" : False,
       "threading" : False 
    }

    global training_target
    #training_target = True 
    training_target = "chb03"
    #training_target = "chb03/chb03_01.edf"



    global csv_path
    csv_path = os.path.abspath("D:\csv-chb-mit")  
    # csv_path = os.path.abspath("../csv-chb-mit-scalp-eeg-database")

    global feature_extraction_path
    feature_extraction_path = "D:\csv-feature-chb-mit"

    global dataset_path
    dataset_path = os.path.abspath("../chb-mit-scalp-eeg-database-1.0.0")

    global warnings_path
    warnings_path = os.path.abspath("../warnings")
    # incriments number on end of file to avoid overwriting other warning files
    existing_files = glob(f"{warnings_path}*")
    if len(existing_files) > 0:
        warnings_path = f"{warnings_path}-{len(existing_files )}" 
