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
import cnn.cnn.generate_datasets

class calculate_metrics:
    def __init__(self):
        models = glob.glob(os.path.join(control.model_save_path, "*", "*.keras"))
        results = glob.glob(os.path.join(control.model_save_path, "*", "*.results"))
        for model in models:
            if model[:-6]+".results" in results:
                print(f"$$ results already exist for {model}")
                continue

            print(f"$$ calculating model metrics for {model}")
            self.train_generator, self.test_generator = cnn.cnn.generate_datasets()

            self.model = load_model(model)

            y_true = self.test_generator.classes
            y_pred = self.model.predict(self.test_generator, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            print("Confusion Matrix:")
            print(cm)
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(classification_report(y_true, y_pred))