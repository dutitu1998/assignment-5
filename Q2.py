import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from file
data = np.loadtxt("ex2data2.txt", delimiter=",")
X = data[:, 0:2]   # First two columns (features)
y = data[:, 2]     # Last column (target)
m = len(y)

# Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Add intercept term
X_b = np.c_[np.ones(m), X_norm]

# Initialize parameters
theta = np.zeros(X_b.shape[1])
alpha = 0.01
iterations = 400

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

# Run gradient descent
theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Learned theta:", theta)
