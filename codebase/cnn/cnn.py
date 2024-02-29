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
from random import sample, shuffle


class cnn:
    def __init__(self, stft_path, batch_size=1, epoch=8):
        self.batch_size = batch_size
        self.epoch = epoch
        self.stft_path = stft_path 
        self.create_model()

        start_time = time.time()
        print(f"start time: {start_time}")
        if control.target == True:
            self.train_all_files()
        elif "/" in control.target:
            print("error")
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


        def train_on_spectogram_batch(paths, label):
            open_files = [np.load(path) for path in paths]
            data = np.concatenate(open_files)  # Concatenate data from all paths
            labels = np.array([one_hot_encoding[label]] * data.shape[0])  # Use data.shape[0] as the number of samples
            self.model.fit(data, labels, batch_size=self.batch_size, epochs=self.epoch)

        def train_on_spectogram(file_path, label):
            data = np.load(file_path)
            labels = np.array([one_hot_encoding[label]] * len(data))
            self.model.fit(data, labels, batch_size=1, epochs=self.epoch)
        
        def predict_spectogram(file_paths, true_label, batch_size=32):
            c = 0
            true_class = "unknown"
            for k, v in one_hot_encoding.items():
                if np.array_equal(v, true_label):
                    true_class = k

            total_loops = (len(file_paths) // self.batch_size)
            for i in range(0, len(file_paths), self.batch_size):
                print(f"[{true_class}] predicting batch {c}/{total_loops}")

                paths = file_paths[i: i + self.batch_size]
                open_files = [np.load(path) for path in paths]
                batch = np.concatenate(open_files)
                predictions = self.model.predict(batch)
                for prediction in predictions:
                    row = np.argmax(true_label)
                    col = np.argmax(prediction)
                    self.confusion_matrix[row][col] += 1

                c += 1


        def train_model():
            c = 0 
            for i in range(0, len(train_ictal_files), self.batch_size):
                print(f"[ictal] training batch {c}/19")
                batch = train_ictal_files[i: i + self.batch_size]
                train_on_spectogram_batch(batch, "ictal")
                c += 1

            c = 0 
            for i in range(0, len(train_preictal_files), self.batch_size):
                print(f"[preictal] training batch {c}/19")
                batch = train_preictal_files[i: i + self.batch_size]
                train_on_spectogram_batch(batch, "preictal")
                c += 1

            c = 0 
            for i in range(0, len(train_interictal_files), self.batch_size):
                print(f"[interictal] training batch {c}/19")
                batch = train_interictal_files[i: i + self.batch_size]
                train_on_spectogram_batch(batch, "interictal")
                c += 1
            self.model.save("complex_trained_model.h5")

        def load_model():
            self.model = tf.keras.models.load_model("trained_model.h5")

        def compute_confusion_matrix():
            self.confusion_matrix = [[0,0,0], [0,0,0], [0,0,0]]
            predict_spectogram(test_ictal_files, one_hot_encoding["ictal"])
            predict_spectogram(test_preictal_files, one_hot_encoding["preictal"])
            predict_spectogram(test_interictal_files, one_hot_encoding["interictal"])
            print("            interictal preictal ictal")
            print(f"interictal      {self.confusion_matrix[0][0]}        {self.confusion_matrix[0][1]}     {self.confusion_matrix[0][2]}")
            print(f"preictal        {self.confusion_matrix[1][0]}        {self.confusion_matrix[1][1]}     {self.confusion_matrix[1][2]}")
            print(f"ictal           {self.confusion_matrix[2][0]}        {self.confusion_matrix[2][1]}     {self.confusion_matrix[2][2]}")




        ictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "ictal", "*"))
        preictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "preictal", "*"))
        interictal_files = glob.glob(os.path.join(control.stft_extraction_path, file_path, "interictal", "*"))

        shuffle(ictal_files)
        shuffle(preictal_files)
        shuffle(interictal_files)

        train_ictal_files, test_ictal_files = self.split_array(ictal_files)
        train_preictal_files, test_preictal_files = self.split_array(preictal_files)
        train_interictal_files, test_interictal_files = self.split_array(interictal_files)


        train_model()
        # load_model()  
        compute_confusion_matrix()


    def split_array(self, array):
        test_element_count = int(len(array) * control.test_percentage)
        test_elements = sample(array, test_element_count)
        train_elements = [item for item in array if item not in test_elements]
        return train_elements, test_elements


    def create_model(self):
        def simple_model():
            model = models.Sequential()

            # Add Convolutional layers
            model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(3841, 2)))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Flatten())

            # Add Dense layers
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(3, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model = model

        def complex_model():
            model = models.Sequential()

            # Add Convolutional layers
            model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(3841, 2)))
            model.add(layers.BatchNormalization())

            model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
            model.add(layers.BatchNormalization())

            model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
            model.add(layers.BatchNormalization())
                      
            model.add(layers.MaxPooling1D(pool_size=2))

            model.add(layers.Flatten())

            # Add Dense layers
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(3, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.model = model

        # simple_model()
        complex_model()



        # # Assuming the input has shape (batch_size, time_steps, n_channels)
        # input_shape = (3841, 2)

        # self.model = models.Sequential()

        # # First convolutional block
        # self.model.add(layers.Conv3D(16, 3, strides=(1, 2), input_shape=input_shape, activation='relu'))
        # self.model.add(layers.BatchNormalization())

        # # Second convolutional block
        # self.model.add(layers.Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation='relu'))
        # self.model.add(layers.BatchNormalization())

        # # Third convolutional block
        # self.model.add(layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu'))
        # self.model.add(layers.BatchNormalization())

        # # MaxPooling layer
        # self.model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

        # # Flatten the output before fully connected layers
        # self.model.add(layers.Flatten())

        # # Fully connected layers
        # self.model.add(layers.Dense(256, activation='relu'))
        # self.model.add(layers.Dropout(0.5))
        # self.model.add(layers.Dense(2, activation='softmax'))

        # # Compile the model
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # # Display the model summary
        # self.model.summary()



        # # Define the model
        # model = models.Sequential()

        # # Convolutional Block 1
        # model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        # model.add(layers.BatchNormalization())

        # # MaxPooling Layer
        # model.add(layers.MaxPooling2D(pool_size=(2,2)))

        # # Flatten Layer
        # model.add(layers.Flatten())

        # # Fully Connected Layers
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(10, activation='softmax'))  # Adjust the output size as needed

        # # Compile the model (you can customize the optimizer, loss, and metrics)
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # # Print the model summary
        # model.summary()
        # self.model = model

  