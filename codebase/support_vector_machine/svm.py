from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocessing(dataset):
    pass



def train_model(features, labels):
    # Assuming you have already loaded and preprocessed your data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create an SVM model
    svm_model = SVC(kernel='linear', C=1.0, gamma='scale')

    # Train the model
    svm_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
