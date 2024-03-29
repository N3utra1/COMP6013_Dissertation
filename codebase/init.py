import control 
import os
from load_dataset.CHB import CHB 
from extract_data.Writer import Writer 
from extract_data.Extractor import Extractor 
import subprocess
import traceback
args = None

def load_dataset():
    chb_mit_path = control.dataset_path
    loaded_chb = []
    if control.target == True: # all patients
        for dir in os.listdir(chb_mit_path):
            if os.path.isdir(os.path.join(chb_mit_path, dir)) and not (dir == "chb24"):
                chb = CHB(dir_name=dir, chb_mit_path=chb_mit_path, verbosity=True) 
                loaded_chb.append(chb)
        return loaded_chb 

    if type(control.target) == str: # a specific patient
        if os.path.isdir(os.path.join(chb_mit_path, control.target)):
            chb = CHB(dir_name=control.target, chb_mit_path=chb_mit_path, verbosity=True) 
            loaded_chb.append(chb)
        return loaded_chb

    print("please set control.target to a valid option (try 'chb06' or True)") 


def write_dataset(loaded_chb):
    for chb in loaded_chb:
        Writer(chb=chb, write_path=control.csv_path, overwrite=control.extractor["overwrite"], threading=control.extractor["threading"])

def extract_features(loaded_chb):
    for chb in loaded_chb:
        Extractor(chb_metadata=chb, csv_path=control.csv_path, write_path=control.stft_extraction_path, overwrite=control.extractor["overwrite"], threading=control.extractor["threading"])

def train_model():
    # tuning, loading and training are all covered by this function
    from cnn.cnn import cnn 
    c = cnn(control.stft_extraction_path, 
                    num_conv_layers=control.single_model_params["num_conv_layers"], 
                    num_dense_layers=control.single_model_params["num_dense_layers"],
                    dense_layer_size=control.single_model_params["dense_layer_size"],
                    epochs=control.single_model_params["epochs"],
                    batch_size=control.single_model_params["batch_size"])

def tune_model():
    print("$$ starting model tuning")
    for conv in control.hyperparam_limits["model_parameters"]["num_conv_layers"]:
        for dense in control.hyperparam_limits["model_parameters"]["num_dense_layers"]:
            for dense_size in control.hyperparam_limits["model_parameters"]["dense_layer_size"]:
                for epoch in control.hyperparam_limits["training_parameters"]["epoch"]:
                    for batch_size in control.hyperparam_limits["training_parameters"]["batch_size"]:
                        try:
                            subprocess.run(["python", 
                                            "./cnn/cnn.py", 
                                            "--conv", str(conv), 
                                            "--dense", str(dense), 
                                            "--dense-size", str(dense_size), 
                                            "--epochs", str(epoch), 
                                            "--batch-size", 
                                            str(batch_size)], check=True)
                        except AttributeError as e:
                            control.warning(f"\n\n $$ target subjects: {control.target} ;\n traceback for error while tuning {conv}.{dense}.{dense_size}.{epoch}.{batch_size}\n")
                            control.warning(traceback.format_exec())
                            continue 

def calculate_metrics():
    from cnn.calculate_metrics import calculate_metrics
    c = calculate_metrics()


def main():
    # load datasets into OOP Objects
    loaded_chb = load_dataset() 

    # write data to disc as CSV files (feature extraction)
    if control.write_dataset:
        print("$$ extracting csv files from dataset")
        write_dataset(loaded_chb)

    # converts the written data to stft windows
    if control.extract_features:
        print("$$ extracting stft windows")
        extract_features(loaded_chb)

    # trains a single model with the structure
    if control.train_single_model:
        print("$$ training model")
        train_model()

    # saves .keras models and saves them in the results dir
    if control.tune_model:
        print("$$ tuning model")
        tune_model()

    # tests the .keras models in the results dir and writes its "results" file
    if control.calculate_metrics:
        print("$$ calculating metrics")
        calculate_metrics()

if __name__ == "__main__":
    control.init()
    main()
