import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_sample = X.shape[0]
    
    for i in range(n_iter):
        index = i % n_sample

        x = X[index]
        actual = y[index]

        pred = np.dot(x,weights)

        error = pred - actual

        gradient = 2*error*x 

        weights = weights - learning_rate*gradient

    return weights.tolist()
