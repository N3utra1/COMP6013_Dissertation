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



class ann:
    def __init__(self, csv_path, batch_size=1000):
        self.csv_path = csv_path        
        print("creating model")

        # Assuming the input has shape (batch_size, time_steps, n_channels)
        input_shape = (time_steps, n_channels)

        model = models.Sequential()

        # First convolutional block
        model.add(layers.Conv3D(16, (1, 5, 5), strides=(1, 2, 2), input_shape=input_shape, activation='relu'))
        model.add(layers.BatchNormalization())

        # Second convolutional block
        model.add(layers.Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation='relu'))
        model.add(layers.BatchNormalization())

        # Third convolutional block
        model.add(layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu'))
        model.add(layers.BatchNormalization())

        # MaxPooling layer
        model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

        # Flatten the output before fully connected layers
        model.add(layers.Flatten())

        # Fully connected layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Display the model summary
        model.summary()



        if control.target == True:
            self.train_all_files()
        elif "/" in control.target:
            self.train_on_file(os.path.join(self.csv_path, control.target))
        else:
            self.train_subject(control.target)   


        start_time = time.time()

        end_time = time.time()

        print("model trained")
        self.model.save("incrimental_learning_model_1.h5")
        print(f"start time: {start_time}")
        print(f"end time: {end_time}")
        self.print_model_metrics()

    def train_all_files(self):
        for chb_dir in os.listdir(self.csv_path):
            sub_path = os.path.join(self.csv_path, chb_dir)
            for chb in os.listdir(sub_path):
                if chb.endswith(".csv"):
                    print(f"reading {chb}")
                file_path = os.path.join(sub_path, chb)
                self.train_on_file(file_path)

    def train_subject(self, subject_dir):
        subjects_files = glob.glob(os.path.join(self.csv_path, subject_dir, "*.csv"))
        print(subjects_files)
        for file in subjects_files:
            self.train_on_file(file)


    def train_on_file(self, file_path):
        df = pd.read_csv(file_path)

        print(f"preparing data for training")
        self.features = df[control.common_columns]
        self.labels = df["Preictal"]

        split_features_and_labels = self.split()
        self.features_train = split_features_and_labels[0]
        self.features_test = split_features_and_labels[1]
        self.labels_train = split_features_and_labels[2]
        self.labels_test = split_features_and_labels[3]

        scaled_results = self.scale()
        self.features_train_scaled = scaled_results[0]
        self.features_test_scaled = scaled_results[1]

        print(f"fitting model for {file_path}")
        self.model.fit(self.features_train_scaled, self.labels_train, epochs=5)

        prediction_value = self.model.predict(self.features_test)
        cmatrix = confusion_matrix(self.labels_test, np.round(prediction_value))
        print(f"cmatrix:\n{cmatrix}")



    def load_CSVs(self, path):

        all_data = pd.DataFrame()
        for chb_dir in os.listdir(path):
            sub_path = os.path.join(path, chb_dir)
            for chb in os.listdir(sub_path):
                if chb.endswith(".csv"):
                    file_path = os.path.join(sub_path, chb)
                    df = pd.read_csv(file_path)
                    all_data = pd.concat([all_data, df], ignore_index=True)
        return all_data
    

    def extract(self):
        return (self.loaded_dataset.iloc[1:, 1:], self.loaded_dataset.iloc[1:, 0])

    def split(self):
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=control.seed)

    def scale(self):
        sk_scaler = StandardScaler()
        scaled_features_train = sk_scaler.fit_transform(self.features_train)
        scaled_features_test = sk_scaler.transform(self.features_test)  
        return (scaled_features_train, scaled_features_test)

    def train_model(self):
        svm_model = LinearSVC(random_state=control.seed, dual="auto")
        svm_model.fit(self.features_train_scaled, self.labels_train)
        return svm_model

    def print_model_metrics(self ):
        prediction = self.model.predict(self.features_test_scaled)
        accuracy = accuracy_score(self.labels_test, prediction)
        report = classification_report(self.labels_test, prediction)
        print(f"Accuracy: {accuracy}")
        print("Report:")
        print(report)