import numpy as np
import matplotlib.pyplot as plt

# ==================== 1. Generate Random Dataset ====================
np.random.seed(42)
m = 100  # Number of samples

# Generate two exam scores (feature1: 30-100, feature2: 30-100)
exam1 = np.random.randint(30, 100, size=m)
exam2 = np.random.randint(30, 100, size=m)

# Admission decision (0 or 1) based on a non-linear boundary
# Let's assume admission = 1 if 0.5*exam1 + 0.7*exam2 > 80
admission = (0.5*exam1 + 0.7*exam2 + 5*np.random.randn(m) > 80).astype(int)

X = np.column_stack((exam1, exam2))
y = admission

# ==================== 2. Visualize Data ====================
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Not Admitted', marker='x')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Admitted', marker='o')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Admission Decisions Based on Exam Scores')
plt.legend()
plt.grid(True)
plt.show()

# ==================== 3. Feature Normalization ====================
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

X_norm, X_mean, X_std = normalize_features(X)
X_b = np.c_[np.ones(m), X_norm]  # Add intercept term

# ==================== 4. Sigmoid Function ====================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==================== 5. Cost Function ====================
def compute_cost(X, y, theta):
    h = sigmoid(X @ theta)
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# ==================== 6. Gradient Descent ====================
def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Initialize parameters
theta = np.zeros(X_b.shape[1])
alpha = 0.1
iterations = 1000

# Run gradient descent
theta, cost_history = gradient_descent(X_b, y, theta, alpha, iterations)

print("Learned theta:", theta)

# ==================== 7. Plot Cost Over Time ====================
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost J(θ)')
plt.title('Cost Function Over Iterations')
plt.grid(True)
plt.show()

# ==================== 8. Decision Boundary ====================
def plot_decision_boundary(X, y, theta, mean, std):
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Not Admitted', marker='x')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Admitted', marker='o')
    
    # Plot decision boundary (where theta^T * x = 0)
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),np.linspace(x2_min, x2_max, 100))
    
    # Normalize the grid points using the same mean and std
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]
    X_grid_norm = (X_grid - mean) / std
    X_grid_b = np.c_[np.ones(X_grid_norm.shape[0]), X_grid_norm]
    
    # Predict probabilities
    probs = sigmoid(X_grid_b @ theta).reshape(xx1.shape)
    
    # Plot decision boundary at p=0.5
    plt.contour(xx1, xx2, probs, levels=[0.5], colors='green')
    
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Decision Boundary for Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, theta, X_mean, X_std)

# ==================== 9. Evaluate Accuracy ====================
def predict(X, theta, mean, std):
    X_norm = (X - mean) / std
    X_b = np.c_[np.ones(X.shape[0]), X_norm]
    return (sigmoid(X_b @ theta) >= 0.5).astype(int)

predictions = predict(X, theta, X_mean, X_std)
accuracy = np.mean(predictions == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")