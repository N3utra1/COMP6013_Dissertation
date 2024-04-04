import pandas as pd
import numpy as np
import glob
import os
from random import sample, shuffle
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K 
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import threading
import argparse

script_dir = os.path.dirname(os.path.realpath(os.path.join(__file__, "..")))
if script_dir not in sys.path:
    sys.path.append(script_dir)
import control
import control
from cnn.cnn import Generator

class calculate_metrics:
    def __init__(self, model_path):
        self.classes = ["interictal", "preictal", "ictal"]
        self.stft_path = control.stft_extraction_path
        self.get_metrics(model_path)

    def get_metrics(self, model):
            print(f"$$ calculating model metrics for {model}")

            self.generator = self.get_generator()
            self.model = load_model(model)

            y_true = self.classes
            y_pred = self.model.predict(self.generator, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            print("Confusion Matrix:")
            print(cm)
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(classification_report(y_true, y_pred))



    def get_generator(self):
        # generate one hot encoded matrix
        integer_mapping = {x: i for i,x in enumerate(self.classes)}
        labels = [integer_mapping[c] for c in self.classes]
        one_hot_matrix = to_categorical(labels, num_classes=len(self.classes))

        mode = type(control.target)
        if  mode == type([]): target_subjects = control.target # already an array
        elif mode == type(bool): target_subjects = [path.split(os.sep)[-1] for path in glob.glob(self.stft_path+os.sep+"*")] # get all subject folders
        elif mode == type(""): target_subjects = [control.target] # only a string
        else: raise ValueError

        file_paths = []
        labels = []
        for subject in target_subjects:
            file_paths += glob.glob(os.path.join(self.stft_path, subject, "*", "*.npy"))
            labels += [one_hot_matrix[self.classes.index(path.split(os.sep)[-2])] for path in file_paths]

        return Generator(file_paths, labels, control.metric_batch_size)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    control.init()
    
    model = calculate_metrics()
