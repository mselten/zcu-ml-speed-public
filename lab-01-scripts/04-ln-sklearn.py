# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Create a binary target variable (threshold at 100)
target_threshold = 100
y_binary = (diabetes.target > target_threshold).astype(int)

# Define features and target
X = diabetes.data
y = y_binary

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standard scaling and polynomial feature engineering
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(n_estimators=125, learning_rate=0.1, verbose=3))
])

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)


train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
print(f"Train Accuracy: {train_accuracy:.3f}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")

# Print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)