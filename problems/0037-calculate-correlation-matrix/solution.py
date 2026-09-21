import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.array(X)

    if Y is None:
        return np.corrcoef(X, rowvar=False)

    Y = np.array(Y)

    result = np.zeros((X.shape[1], Y.shape[1]))

    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            result[i, j] = np.corrcoef(X[:, i], Y[:, j])[0, 1]

    return result