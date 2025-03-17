import numpy as np
import matplotlib.pyplot as plt

# Define the number of data points
n_data = 20

# Generate x values
x = np.linspace(-100, 100, n_data)

# Define a more complex function (1D)
def func(x):
    return 2 * np.sin(x) + 3 * x**9 - 4 * x  - 5 * x**2

# Generate y values using the function
y_true = func(x)

# Add some noise to y values
np.random.seed(42)  # For reproducibility
noise = np.random.normal(loc=0, scale=5, size=n_data)
y_noisy = y_true + noise

# Choose the degree N for polynomial feature engineering
N = 3  # Change this value to experiment with different degrees

# Create a design matrix X with polynomial features of degree N
X = np.zeros((n_data, N + 1))
for i in range(N + 1):
    X[:, i] = x ** i

# Explicit linear regression calculation
weights = np.linalg.inv(X.T @ X) @ X.T @ y_noisy  # (X^T * X)^-1 * X^T * y

# Print the estimated weights
print("Estimated weights:", weights)

# Generate predicted values using the estimated weights
y_pred = X @ weights

# Plot the data and the fitted line
plt.scatter(x, y_noisy, label="Noisy Data")
plt.plot(np.linspace(-100, 100, 400), func(np.linspace(-100, 100, 400)), 'r', label="True Function")
plt.plot(x, y_pred, 'g', label=f"Fitted Line (Degree {N})")
plt.legend()
plt.show()