import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Feature Mapping
def map_features(x1, x2, degree=6):
    out = [np.ones(x1.shape[0])]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j)) * (x2 ** j))
    return np.stack(out, axis=1)

# Step 2: Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Cost Function with Regularization
def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + reg

# Step 4: Gradient with Regularization
def gradient_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1 / m) * (X.T @ (h - y))
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    return grad

# Step 5: Predict
def predict(theta, X):
    return sigmoid(X @ theta) >= 0.5

# Step 6: Plot Decision Boundary
def plot_decision_boundary(theta, X, y, lambda_):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_features(np.array([u[i]]), np.array([v[j]])).dot(theta)
    z = z.T

    plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    plot_data(X[:, 1:3], y)
    plt.title(f"Decision Boundary (lambda = {lambda_})")
    plt.show()

# Step 7: Plot Data
def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+', label='Pass')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o', label='Fail')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()

# Step 8: Generate Synthetic Data
def generate_data():
    np.random.seed(0)
    num = 100
    x1 = np.random.normal(0, 1, num)
    x2 = np.random.normal(0, 1, num)
    y = (x1**2 + x2**2 + 0.3*np.random.randn(num)) < 1.5
    return np.column_stack((x1, x2)), y.astype(int)

# Step 9: Main Execution
X_raw, y = generate_data()
X_mapped = map_features(X_raw[:, 0], X_raw[:, 1])
initial_theta = np.zeros(X_mapped.shape[1])

# Try different lambda values
for lambda_ in [0, 1, 100]:
    result = minimize(
        fun=cost_function_reg,
        x0=initial_theta,
        args=(X_mapped, y, lambda_),
        method='TNC',
        jac=gradient_reg
    )
    theta_opt = result.x
    plot_decision_boundary(theta_opt, X_mapped, y, lambda_)
    predictions = predict(theta_opt, X_mapped)
    accuracy = np.mean(predictions == y) * 100
    print(f"Accuracy with lambda = {lambda_}: {accuracy:.2f}%")
