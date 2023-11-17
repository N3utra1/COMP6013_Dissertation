import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import os
import time



class svm_linear_model:
    def __init__(self, csv_path):
        # load dataset
        print("loading dataset")
        self.loaded_dataset = self.load_CSVs(csv_path) 
        print("dataset loaded")

        #extract features and labels
        extracted = self.extract()
        self.features = extracted[0]
        self.labels = extracted[1]
        print("extracted features")

        # split dataset into testing and training
        split_features_and_labels = self.split()
        self.features_train = split_features_and_labels[0]
        self.features_test = split_features_and_labels[1]
        self.labels_train = split_features_and_labels[2]
        self.labels_test = split_features_and_labels[3]
        print("split dataset")

        # scale training set
        scaled_results = self.scale()
        self.features_train_scaled = scaled_results[0]
        self.features_test_scaled = scaled_results[1]
        print("scaled dataset")

        # train model        
        print("training_model")
        start_time = time.time()
        model = self.train_model()
        end_time = time.time()
        print("model trained")
        print(f"start time: {start_time}")
        print(f"end time: {end_time}")
        self.print_model_metrics(model)


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
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=101)

    def scale(self):
        sk_scaler = StandardScaler()
        scaled_features_train = sk_scaler.fit_transform(self.features_train)
        scaled_features_test = sk_scaler.transform(self.features_test)  
        return (scaled_features_train, scaled_features_test)

    def train_model(self):
        svm_model = LinearSVC(random_state=101, dual="auto")
        svm_model.fit(self.features_train_scaled, self.labels_train)
        return svm_model

    def print_model_metrics(self, model):
        prediction = model.predict(self.features_test_scaled)
        accuracy = accuracy_score(self.labels_test, prediction)
        report = classification_report(self.labels_test, prediction)
        print(f"Accuracy: {accuracy}")
        print("Report:")
        print(report)



class svm_model():
    def __init__(self, csv_path):
        # load dataset
        print("loading dataset")
        self.loaded_dataset = self.load_CSVs(csv_path) 
        print("dataset loaded")

        #extract features and labels
        extracted = self.extract()
        self.features = extracted[0]
        self.labels = extracted[1]
        print("extracted features")

        # split dataset into testing and training
        split_features_and_labels = self.split()
        self.features_train = split_features_and_labels[0]
        self.features_test = split_features_and_labels[1]
        self.labels_train = split_features_and_labels[2]
        self.labels_test = split_features_and_labels[3]
        print("split dataset")

        # scale training set
        scaled_results = self.scale()
        self.features_train_scaled = scaled_results[0]
        self.features_test_scaled = scaled_results[1]
        print("scaled dataset")

        # train model        
        print("training_model")
        start_time = time.time()
        model = self.train_model()
        end_time = time.time()
        print("model trained")
        print(f"start time: {start_time}")
        print(f"end time: {end_time}")
        self.print_model_metrics(model)


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
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=101)

    def scale(self):
        sk_scaler = StandardScaler()
        scaled_features_train = sk_scaler.fit_transform(self.features_train)
        scaled_features_test = sk_scaler.transform(self.features_test)  
        return (scaled_features_train, scaled_features_test)

    def train_model(self):
        svm_model = SVC(kernel="linear")
        svm_model.fit(self.features_train_scaled, self.labels_train)
        return svm_model

    def print_model_metrics(self, model):
        prediction = model.predict(self.features_test_scaled)
        accuracy = accuracy_score(self.labels_test, prediction)
        report = classification_report(self.labels_test, prediction)
        print(f"Accuracy: {accuracy}")
        print("Report:")
        print(report)



