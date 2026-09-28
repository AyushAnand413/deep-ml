import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    predictions = X_batch @ weights + bias

    error = predictions - y_batch

    m = len(batch_indices)

    gradient_weights = (2 / m) * (X_batch.T @ error)

    gradient_bias = (2 / m) * np.sum(error)

    weights = weights - lr * gradient_weights
    bias = bias - lr * gradient_bias

    return np.append(weights, bias)