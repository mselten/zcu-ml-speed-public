# MODEL = 1: Simple Logistic Regression
# MODEL = 2: Neural Network (MLPClassifier)
# MODEL = 3: Boosted Decision Trees (AdaBoostClassifier)
# MODEL = 4: Ensemble of three Neural Networks

MODEL = 1

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the iris dataset
digit = datasets.load_digits()
X = digit.data
y = digit.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

if MODEL == 1:
    # Simple Logistic Regression
    model = LogisticRegression(max_iter=1000)
elif MODEL == 2:
    # Neural Network (MLPClassifier)
    model = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1000)
elif MODEL == 3:
    # Boosted Decision Trees (AdaBoostClassifier)
    model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
elif MODEL == 4:
    # Ensemble of three Neural Networks
    model1 = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1000)
    model2 = MLPClassifier(hidden_layer_sizes=(30, 10), max_iter=1000)
    model3 = MLPClassifier(hidden_layer_sizes=(40, 15), max_iter=1000)
    model = VotingClassifier(estimators=[('model1', model1), ('model2', model2), ('model3', model3)], voting='soft')
else:
    print("Invalid model choice")
    exit()

# Train the model
if MODEL == 4:
    for estimator in model.estimators:
        estimator[1].fit(X_train_std, y_train)
else:
    model.fit(X_train_std, y_train)

# Make predictions on test set
if MODEL == 4:
    predictions = model.predict(X_test_std)
else:
    predictions = model.predict(X_test_std)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
