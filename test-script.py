# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the diabetes dataset from scikit-learn
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create a Pandas DataFrame to work with the data
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

# Print some basic statistics about the data
print("Mean of the features:", np.mean(X, axis=0))
print("Standard deviation of the features:", np.std(X, axis=0))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train a small neural network with one hidden layer
mlp_model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

# Calculate the mean squared error for both models
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"Linear Regression MSE: {mse_lr:.2f}")
print(f"Neural Network MSE: {mse_mlp:.2f}")

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred_lr, label="Linear Regression")
plt.scatter(y_test, y_pred_mlp, label="Neural Network")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Plot a histogram of the predicted values
plt.hist(y_pred_lr, bins=20, alpha=0.5, label="Linear Regression")
plt.hist(y_pred_mlp, bins=20, alpha=0.5, label="Neural Network")
plt.xlabel("Predicted Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()