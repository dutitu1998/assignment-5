import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

# Step 1: Generate classification data
def generate_data(n=118, noise=0.05, seed=42):
    np.random.seed(seed)
    X = np.random.rand(n, 2) * 2 - 1
    y = ((0.2**2 < (X**2).sum(1)) & ((X**2).sum(1) < 0.7**2)) | (((X[:,0]-0.2)**2 + (X[:,1]+0.4)**2) < 0.3**2)
    y = y.astype(int)
    if noise:
        flip = np.random.choice(n, int(n*noise), replace=False)
        y[flip] ^= 1
    return X, y

# Step 2: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Cost and gradient
def cost(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-7, 1 - 1e-7)
    reg = lam / (2*m) * np.sum(theta[1:]**2)
    return -np.mean(y*np.log(h)+(1-y)*np.log(1-h)) + reg

def gradient(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    grad[1:] += (lam / m) * theta[1:]
    return grad

# Step 4: Plot decision boundary
def plot_boundary(X, y, mapper, theta, lam, ax):
    ax.scatter(*X[y==1].T, c='blue', edgecolors='k', label='Pass')
    ax.scatter(*X[y==0].T, c='red', marker='x', label='Fail')
    u = np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 200)
    v = np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1, 200)
    U, V = np.meshgrid(u, v)
    Z = np.array([mapper.transform([[a, b]]) @ theta for a, b in zip(U.ravel(), V.ravel())]).reshape(U.shape)
    ax.contour(U, V, Z, levels=[0], colors='green')
    ax.set_title(f"λ = {lam}")
    ax.grid(True)
    ax.legend()

# Main
if __name__ == "__main__":
    X, y = generate_data(seed=10)
    mapper = PolynomialFeatures(6)
    X_poly = mapper.fit_transform(X)
    lambdas = [0, 1, 5]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, lam in enumerate(lambdas):
        result = minimize(cost, np.zeros(X_poly.shape[1]), args=(X_poly, y, lam),
                          jac=gradient, method='TNC', options={'maxiter': 400})
        theta = result.x
        preds = sigmoid(X_poly @ theta) >= 0.5
        acc = (preds == y).mean() * 100
        plot_boundary(X, y, mapper, theta, lam, axs[i])
        axs[i].text(0.05, 0.05, f'Acc: {acc:.2f}%', transform=axs[i].transAxes,
                    bbox=dict(fc='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig("logistic_plot.png")
    plt.show()
