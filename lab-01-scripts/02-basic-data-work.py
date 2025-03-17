import numpy as np
import matplotlib.pyplot as plt

# Create a 2D array (matrix) using NumPy
np.random.seed(0)
matrix = np.random.rand(3, 4)

print("Matrix:")
print(matrix)

# Accessing elements of the matrix
print("\nAccessing element at position [1, 2]:")
print(matrix[1, 2])

# Modify an element in the matrix
matrix[1, 2] = 10
print("\nModified matrix:")
print(matrix)

# Calculating sum and mean of each row/column
row_sums = np.sum(matrix, axis=1)
col_sums = np.sum(matrix, axis=0)
mean_row = np.mean(matrix, axis=1)
mean_col = np.mean(matrix, axis=0)

print("\nRow sums:")
print(row_sums)
print("Column sums:")
print(col_sums)
print("Mean row values:")
print(mean_row)
print(f"Mean column values: {mean_col}")

# Reshaping a matrix
matrix_reshaped = matrix.reshape(12, 1)
print("\nReshaped matrix:")
print(matrix_reshaped)

# Transposing a matrix
matrix_transpose = np.transpose(matrix)
print("\nTransposed matrix:")
print(matrix_transpose)

# Creating a scatter plot using Matplotlib
np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c='blue', alpha=0.5)
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Creating a line plot using Matplotlib
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(10, 8))
plt.plot(x, y, c='red', lw=2)
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Creating a bar plot using Matplotlib
x = np.arange(1, 6)
y = [3, 5, 1, 4, 7]
plt.figure(figsize=(10, 8))
plt.bar(x, y, c='green', alpha=0.5)
plt.title("Bar Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()