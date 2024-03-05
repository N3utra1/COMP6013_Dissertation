import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import numpy as np

import glob
import os
import time
import control
import datetime
from random import sample, shuffle


class cnn:
    def __init__(self, stft_path, num_conv_layers=4, num_dense_layers=4, dense_layer_size=64):
        self.stft_path = stft_path 
        self.create_model(num_conv_layers=num_conv_layers, num_dense_layers=num_dense_layers, dense_layer_size=dense_layer_size)

        start_time = time.time()
        print(f"start time: {start_time}")
        if control.target == True:
            self.train_all_files()
        elif "/" in control.target:
            print("training on a specific file is not supported. Please specify either a subject or set control.target to True")
            raise RuntimeError
        else:
            self.train_on_subject(control.target)   
        end_time = time.time()
        print("finished all training model trained")
        print(f"end time: {end_time}")


    def train_all_files(self):
        subjects = [path.split(os.sep)[-1] for path in glob.glob(os.path.join(self.stft_path), "*")]
        for subject in subjects:
            self.train_on_subject(subject)

    def train_on_subject(self, file_path):
        one_hot_encoding = {
            "interictal" : np.array([1,0,0]),
            "preictal" : np.array([0,1,0]),
            "ictal" : np.array([0,0,1])
        }


        def predict_spectogram(file_paths, true_label, batch_size=32):
            c = 0
            total_loops = (len(file_paths) // batch_size)
            for i in range(0, len(file_paths), batch_size):
                print(f"[{true_label}] predicting batch {c}/{total_loops}")
                paths = file_paths[i: i + batch_size]
                open_files = np.array([np.load(path) for path in paths])
                batch = np.reshape(np.array(open_files), (len(open_files), 17, 3841, 2))
                predictions = self.model.predict(batch)
                for prediction in predictions:
                    row = np.argmax(one_hot_encoding[true_label])
                    col = np.argmax(prediction)
                    self.confusion_matrix[row][col] += 1
                c += 1

        def train_model(batch_size=8, epochs=1):
            def train_on_spectogram(batch):
                data_batch = []
                labels_batch = []

                for one_hot_class, path in batch:
                    data = np.load(path)
                    data_batch.append(data)
                    labels_batch.append(one_hot_class)

                data_batch = np.reshape(data_batch, (len(data_batch), 17, 3841, 2))
                labels_batch = np.reshape(labels_batch, (len(labels_batch), 3))

                self.model.fit(data_batch, labels_batch, batch_size=batch_size, epochs=epochs)

            def save_model():
                # save path structured as:
                # control.model_save_path/num_conv_layers.num_dense_layers.dense_layer_size/batch_size.epochs.mode.keras
                model_directory = f"{self.num_conv_layers}.{self.num_dense_layers}.{self.dense_layer_size}/"
                output_path_prefix = os.path.normpath(os.path.join(control.model_save_path, model_directory))
                if not os.path.exists(output_path_prefix): os.mkdir(output_path_prefix)
                model_output_path = os.path.join(output_path_prefix, f"{batch_size}.{epochs}")
                if os.path.exists(model_output_path):
                        self.model.save(f"{model_output_path}.{datetime.datetime.now().strftime('%d.%m.%Y.%H.%M')}.keras")
                else:
                    self.model.save(f"{model_output_path}.keras")

                print(f"saved model {model_output_path}")

                results = compute_confusion_matrix()
                with open(f"{model_output_path}.results", "w") as file:
                    file.writelines(results)
                print(results)
                print(f"saved results {results}")

            # add the classes to each file
            all_files = []
            [all_files.append([one_hot_encoding["ictal"], path]) for path in train_ictal_files]
            [all_files.append([one_hot_encoding["preictal"], path]) for path in train_preictal_files]
            [all_files.append([one_hot_encoding["interictal"], path]) for path in train_interictal_files]
            shuffle(all_files)

            c = 0 
            total_loops = len(all_files) // batch_size
            for i in range(0, len(all_files), batch_size):
                print(f"training batch {c}/{total_loops}")
                batch = all_files[i: i + batch_size]
                train_on_spectogram(batch)
                c += 1

            save_model()

        def tune_model():
            for e in control.hyperparam_limits["training_parameters"]["epoch"]:
                for b in control.hyperparam_limits["training_parameters"]["batch_size"]:
                    train_model(batch_size=b, epochs=e)

        def load_model():
            self.model = tf.keras.models.load_model(control.load_model)

        def compute_confusion_matrix():
            self.confusion_matrix = [[0,0,0], [0,0,0], [0,0,0]]
            predict_spectogram(test_ictal_files, "ictal")
            predict_spectogram(test_preictal_files, "preictal")
            predict_spectogram(test_interictal_files, "interictal")
            return f"""            interictal preictal ictal
            interictal      {self.confusion_matrix[0][0]}        {self.confusion_matrix[0][1]}     {self.confusion_matrix[0][2]}
            preictal        {self.confusion_matrix[1][0]}        {self.confusion_matrix[1][1]}     {self.confusion_matrix[1][2]}
            ictal           {self.confusion_matrix[2][0]}        {self.confusion_matrix[2][1]}     {self.confusion_matrix[2][2]}"""
            




        ictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "ictal", "*"))
        preictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "preictal", "*"))
        interictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "interictal", "*"))

        shuffle(ictal_files)
        shuffle(preictal_files)
        shuffle(interictal_files)

        train_ictal_files, test_ictal_files = self.split_array(ictal_files)
        train_preictal_files, test_preictal_files = self.split_array(preictal_files)
        train_interictal_files, test_interictal_files = self.split_array(interictal_files)


        if control.load_model and (not control.train_model and not control.tune_model):
            load_model()  
        elif control.train_model and (not control.load_model or not control.tune_model):
            train_model(control.batch_size, control.epochs)
        elif control.tune_model and (not control.load_model or not control.train_model):
            tune_model()
        else:
            print("only one of these can be true: \n\tcontrol.load_model\n\ntcontrol.train_model\n\tcontrol.tune_model")
            raise RuntimeError


    def split_array(self, array):
        test_element_count = int(len(array) * control.test_percentage)
        test_elements = sample(array, test_element_count)
        train_elements = [item for item in array if item not in test_elements]
        return train_elements, test_elements


    def create_model(self, num_conv_layers=4, num_dense_layers=4, dense_layer_size=64):
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers 
        self.dense_layer_size = dense_layer_size 

        if num_dense_layers < 1 or num_conv_layers < 0: 
            print("Layer count cannot be less than 1")
            raise ValueError

        model = models.Sequential()

        # add convolutional layers
        for i in range(num_conv_layers):
            model.add(layers.Conv2D(32 * (2**i), kernel_size=(3, 3), activation='relu', input_shape=(17, 3841, 2)))
            model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())

        # add dense layers
        for i in range(num_dense_layers-1):
            model.add(layers.Dense(dense_layer_size, activation='relu'))

        model.add(layers.Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model