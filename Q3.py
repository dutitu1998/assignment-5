import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Random Dataset (You can replace this with real data)
np.random.seed(0)
m = 100  # number of examples
X = np.random.randn(m, 2) * 10 + 50  # exam scores around 50
y = (X[:, 0] + X[:, 1] > 100).astype(int)  # admit if total score > 100

# 2. Visualize data
def plot_data(X, y):
    admitted = y == 1
    not_admitted = y == 0
    plt.scatter(X[admitted, 0], X[admitted, 1], c='b', label='Admitted')
    plt.scatter(X[not_admitted, 0], X[not_admitted, 1], c='r', label='Not Admitted')
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.title("Student Admissions")
    plt.show()

plot_data(X, y)

# 3. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost

# 5. Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient
    return theta

# Add intercept term
X_b = np.c_[np.ones(m), X]
theta = np.zeros(X_b.shape[1])
alpha = 0.01
iterations = 1000

# 6. Train model
theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Learned theta:", theta)

# 7. Plot Decision Boundary
def plot_decision_boundary(X, y, theta):
    plot_data(X[:, 1:], y)  # remove intercept for plotting
    x_vals = np.array([X[:,1].min(), X[:,1].max()])
    y_vals = -(theta[0] + theta[1]*x_vals) / theta[2]
    plt.plot(x_vals, y_vals, 'g-', label='Decision Boundary')
    plt.legend()
    plt.show()

plot_decision_boundary(X_b, y, theta)

# 8. Accuracy
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

predictions = predict(X_b, theta)
accuracy = np.mean(predictions == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")
