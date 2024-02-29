import control 
import os
from load_dataset.CHB import CHB 
from extract_data.Writer import Writer 
from extract_data.Extractor import Extractor 

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

    print("please set control.target to a valid option (try 'chb03' or True)") 



def write_dataset(loaded_chb):
    for chb in loaded_chb:
        Writer(chb=chb, write_path=control.csv_path, overwrite=control.extractor["overwrite"], threading=control.extractor["threading"])

def extract_features(loaded_chb):
    for chb in loaded_chb:
        Extractor(chb_metadata=chb, csv_path=control.csv_path, write_path=control.stft_extraction_path, overwrite=control.extractor["overwrite"], threading=control.extractor["threading"])

def train_model():
    from cnn.cnn import cnn 
    avaliable_models = {
        "cnn" : cnn 
    }
    print(f"selected model {control.model}")
    avaliable_models[control.model](control.stft_extraction_path, batch_size=control.batch_size, epoch=control.epoch)


def main():
    # load datasets into OOP Objects
    if control.load_dataset or control.write_dataset or control.extract_features:
        loaded_chb = load_dataset() 

    # write data to disc as CSV files (feature extraction)
    if control.write_dataset:
        write_dataset(loaded_chb)

    if control.extract_features:
        extract_features(loaded_chb)

    if control.train_model:
        train_model()


if __name__ == "__main__":
    control.init()
    main()