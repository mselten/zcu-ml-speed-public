# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset (digits)
digits = datasets.load_digits()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.85, random_state=42)

# Create a pipeline with StandardScaler and MLPClassifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, verbose=True))
])

# Train the model using the training sets
pipe.fit(X_train, y_train)

# Make predictions on the train set
train_predictions = pipe.predict(X_train)
# Evaluate the model accuracy on train set
train_accuracy = accuracy_score(y_train, train_predictions)
print("Train Accuracy:", train_accuracy)

# Make predictions on the test set
test_predictions = pipe.predict(X_test)
# Evaluate the model accuracy on test set
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)