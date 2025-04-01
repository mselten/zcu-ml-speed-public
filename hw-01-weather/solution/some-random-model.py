INFER_TEST_DATA = True  # Set to True to make predictions on test data

# Choose a model by setting MODEL to 1, 2, 3, or 4
MODEL = 3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Load training data
data = pd.read_csv('data/train-data.csv')

# Preprocess time column
data['time'] = pd.to_datetime(data['time']).dt.hour.astype(float)

# Split data into features (X) and target (y)
X = data.drop('coco', axis=1)
y = data['coco']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Choose a model based on the MODEL variable
if MODEL == 1:
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', verbose=3, n_jobs=-1, max_iter=1000)
    )
elif MODEL == 2:
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(SVC(kernel='linear', probability=True, C=0.1))
    )
elif MODEL == 3:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    )
elif MODEL == 4:
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, verbose=1))
    )

# Train the model
model.fit(X_train, y_train)

# Make predictions on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Get predicted probabilities for training and testing sets
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)


# Evaluate the model's performance
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.1f}%")
train_labels = np.unique(y_train)
print(f"Train F1 Score: {f1_score(y_train, y_train_pred, average='weighted', labels=train_labels) * 100:.1f}%")

y_train_proba_argmax = np.argmax(y_train_proba, axis=1) + 1  # Assuming classes are numbered from 1 to 24
print(f"Train Log Loss: {log_loss(y_train, y_train_proba, labels=np.unique(y_train)):.4f}")

# Test metrics
test_labels = np.unique(y_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.1f}%")
print(f"Test F1 Score: {f1_score(y_test, y_test_pred, average='weighted', labels=test_labels) * 100:.1f}%")


if INFER_TEST_DATA:
    # Load test data
    test_data = pd.read_csv('data/test-data.csv')

    # Preprocess time column
    test_data['time'] = pd.to_datetime(test_data['time']).dt.hour.astype(float)

    # Retrain the model on all training data
    model.fit(X, y)

    # Make predictions on test data
    test_predictions = model.predict(test_data)

    # Save predictions to a file
    with open('data/model-predictions.csv', 'w') as f:
        for prediction in test_predictions:
            f.write(str(prediction) + '\n')