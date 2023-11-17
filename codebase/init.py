import control 
import os
from load_dataset.CHB import CHB 
from extract_data.Extractor import Extractor
from svm.svm import svm_model 
from svm.svm import svm_linear_model 

def load_dataset():
    chb_mit_path = control.dataset_path
    loaded_chb = []
    for dir in os.listdir(chb_mit_path):
        if os.path.isdir(os.path.join(chb_mit_path, dir)) and not (dir == "chb24"):
            chb = CHB(dir_name=dir, chb_mit_path=chb_mit_path, verbosity=True) 
            loaded_chb.append(chb)
    return loaded_chb 

def write_dataset(loaded_chb):
    for chb in loaded_chb:
            Extractor(chb=chb, write_path=control.csv_path, overwrite=control.extractor["overwrite"], threading=control.extractor["threading"])
            break

def train_model():
    avaliable_models = {
        "svm" : svm_model,
        "svm-linear" : svm_linear_model 
    }
    print(f"selected model {control.model}")
    avaliable_models[control.model](control.csv_path)


def main():
    # load datasets into OOP Objects
    if control.load_dataset or control.write_dataset:
        loaded_chb = load_dataset() 

    # write data to disc as CSV files (feature extraction)
    if control.write_dataset:
        write_dataset(loaded_chb)
        

    if control.train_model:
        train_model()


if __name__ == "__main__":
    control.init()
    main()