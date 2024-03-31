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
import control


class Generator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.image_filenames) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        batch_start = i * self.batch_size
        batch_end = (i + 1) * self.batch_size
        batch_files = self.image_filenames[batch_start:batch_end]
        batch_labels = self.labels[batch_start:batch_end]

        batch_images = []
        for file in batch_files:
            try:
                image = np.load(file, allow_pickle=True)
            except Exception as e:
                print(f"error when opening {file}")
                raise e
            batch_images.append(image)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        return batch_images, batch_labels




class calculate_metrics:
    def __init__(self):
        self.classes = ["interictal", "preictal", "ictal"]
        self.stft_path = control.stft_extraction_path
        models = glob.glob(os.path.join(control.model_save_path, "*", "*.keras"))
        results = glob.glob(os.path.join(control.model_save_path, "*", "*.results"))
        for model in models:
            print(f"$$ model {models.index(model)}/{len(models)}")
            results_file = model[:-6]+".results"
            if results_file in results:
                print(f"$$ results already exist for {model}")
                continue
            
            print(f"$$ calculating model metrics for {model}")

            print("$$ creating generator")
            self.generator = self.get_generator()
            print("$$ loading model")
            self.model = load_model(model)
            
            print("$$ making predictions")
            y_true = []
            y_pred = []
            for batch_images, batch_labels in self.generator:
                y_true.append(np.argmax(batch_labels, axis=1))
                y_pred.append(np.argmax(self.model.predict(batch_images), axis=1))
            print("$$ predictions done")
            y_true = [self.classes[value] for value in np.concatenate(y_true)]
            y_pred = [self.classes[value] for value in np.concatenate(y_pred)]

            cm = confusion_matrix(y_true, y_pred, labels=self.classes)
            accuracy = accuracy_score(y_true, y_pred)
            print(f"writing to {results_file} :")
            print("Confusion Matrix:")
            print(cm)
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(classification_report(y_true, y_pred))
            with open(results_file, "w") as file:
                file.write("Confusion Matrix:")
                file.write(str(cm))
                file.write("\n")
                file.write(f"Accuracy: {accuracy}")
                file.write("Classification Report:")
                file.write(classification_report(y_true, y_pred))




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
        print(f"$$ {len(file_paths)} test files")
        return Generator(file_paths, labels, control.metric_batch_size)
        

