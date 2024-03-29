import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K 
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.metrics import confusion_matrix

import numpy as np
import argparse

import glob
import os
import time
import datetime
import threading
from random import sample, shuffle
import sys

script_dir = os.path.dirname(os.path.realpath(os.path.join(__file__, "..")))
if script_dir not in sys.path:
    sys.path.append(script_dir)
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
            image = np.load(file.decode("utf-8"), allow_pickle=True)
            batch_images.append(image)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        return batch_images, batch_labels


class cnn:
    def __init__(self, stft_path, num_conv_layers=4, num_dense_layers=4, dense_layer_size=64, epochs=1, batch_size=64):
        self.stft_path = stft_path
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.dense_layer_size = dense_layer_size 
        self.epochs = epochs 
        self.batch_size = batch_size

        exists = glob.glob(os.path.join(os.path.normpath(os.path.join(control.model_save_path, f"{self.num_conv_layers}.{self.num_dense_layers}.{self.dense_layer_size}/")), f"{self.batch_size}.{self.epochs}.*"))
        if exists: 
            print(f"skipping {self.num_conv_layers}.{self.num_dense_layers}.{self.dense_layer_size}/{self.batch_size}.{self.epochs}")
            return
    
        print(f"""
            $$      current configuration

                conv layers count   :   {self.num_conv_layers}
                    dense layer count   :   {self.num_dense_layers}
                        dense layer size    :   {self.dense_layer_size}

                            epochs  :   {self.epochs}
                                batch_size  :   {self.batch_size}

            $$
                """)

        try:
            physical_devices = tf.config.list_physical_devices('GPU') 
            if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        # get classification and test datasets
        self.generate_datasets()
        # compile model
        self.compile_model(num_conv_layers=num_conv_layers, num_dense_layers=num_dense_layers, dense_layer_size=dense_layer_size)
        # train model
        self.train()

    def generate_datasets(self):
        # generate one hot encoded matrix
        classes = ["interictal", "preictal", "ictal"]
        integer_mapping = {x: i for i,x in enumerate(classes)}
        labels = [integer_mapping[c] for c in classes]
        one_hot_matrix = to_categorical(labels, num_classes=len(classes))

        # generates a array of subject names called target_subjects
        mode = type(control.target)
        if  mode == type([]): target_subjects = control.target # already an array
        elif mode == type(bool): target_subjects = [path.split(os.sep)[-1] for path in glob.glob(self.stft_path+os.sep+"*")] # get all subject folders
        elif mode == type(""): target_subjects = [control.target] # only a string
        else: raise ValueError

        # Combine datasets for each class and shuffle
        file_paths = glob.glob(os.path.join(self.stft_path, "*", "*", "*.npy"))
        labels = [one_hot_matrix[classes.index(path.split(os.sep)[-2])] for path in file_paths]
        # combined_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels)).shuffle(len(file_paths)).batch(self.batch_size)
        combined_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels)).shuffle(32)

        # if more than one subject, take the average number of samples 
        if len(target_subjects) > 1: 
            combined_dataset = combined_dataset.take(len(combined_dataset) // len(target_subjects))
        

        # split into testing and training 
        train_size = int(len(combined_dataset) * control.train_percentage)
        self.train_dataset = combined_dataset.take(train_size)
        self.test_dataset = combined_dataset.skip(train_size) 
        self.test_dataset = self.test_dataset.take(min(1000, len(self.test_dataset)))

        train_paths = []
        train_labels = []
        for path, label in self.train_dataset:
            train_paths.append(path.numpy())  
            train_labels.append(label.numpy())
        train_paths = np.array(train_paths)
        train_labels = np.array(train_labels) 
        self.train_generator = Generator(train_paths, train_labels, self.batch_size)

        test_paths = []
        test_labels = []
        for path, label in self.test_dataset:
            test_paths.append(path.numpy())  
            test_labels.append(label.numpy())
        test_paths = np.array(test_paths)
        test_labels = np.array(test_labels) 
        self.test_generator = Generator(test_paths, test_labels, self.batch_size)


    def compile_model(self, num_conv_layers=4, num_dense_layers=4, dense_layer_size=64):
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers 
        self.dense_layer_size = dense_layer_size 

        if num_dense_layers < 1 or num_conv_layers < 0: 
            print("$$ Layer count cannot be less than 1")
            raise ValueError

        model = models.Sequential()

        # add convolutional layers
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(17, 3841, 2)))
        for i in range(1, num_conv_layers):
            model.add(layers.Conv2D(32 * (2**i), kernel_size=(3, 3), activation='relu'))
            model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())

        # add dense layers
        for i in range(num_dense_layers-1):
            model.add(layers.Dense(dense_layer_size, activation='relu'))

        model.add(layers.Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self):
        self.model.fit(self.train_generator, epochs=self.epochs, batch_size=self.batch_size)

        directory_path = os.path.normpath(os.path.join(control.model_save_path, f"{self.num_conv_layers}.{self.num_dense_layers}.{self.dense_layer_size}/"))
        if not os.path.exists(directory_path): os.mkdir(directory_path)
        model_output_path = os.path.join(directory_path, f"{self.batch_size}.{self.epochs}")
        self.model.save(f"{model_output_path}.keras")

        result = self.model.evaluate(self.test_generator, batch_size=self.batch_size)
        print(result)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv', type=int, default=4)
    parser.add_argument('--dense', type=int, default=4)
    parser.add_argument('--dense-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    control.init()
    
    model = cnn(control.stft_extraction_path, num_conv_layers=args.conv, num_dense_layers=args.dense, dense_layer_size=args.dense_size, epochs=args.epochs, batch_size=args.batch_size)
