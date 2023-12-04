import os

def init():
    global load_dataset
    load_dataset = False
    global write_dataset 
    write_dataset = False
    global train_model 
    train_model = True 


    global avaliable_models
    avaliable_models = [
       "svm"
       "svm-linear"
    ]
       
    global model 
    model = "svm-linear"

    global extractor
    extractor = {
       "overwrite" : True,
       "threading" : True 
    }

    global csv_path
    csv_path = os.path.abspath("D:\csv-chb-mit")  
    # csv_path = os.path.abspath("../csv-chb-mit-scalp-eeg-database")

    global dataset_path
    dataset_path = os.path.abspath("../chb-mit-scalp-eeg-database-1.0.0")