import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0]  # Population (feature)
y = data[:, 1]  # Profit (target)
m = len(y)

# Add intercept term to X
X_b = np.c_[np.ones(m), X]  # Add bias term (column of ones)

theta = np.zeros(2)
alpha = 0.01
iterations = 1500

def compute_cost(X, y, theta):
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)

def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Learned theta:", theta)
