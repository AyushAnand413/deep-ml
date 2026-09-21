import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    matrix = np.array(matrix)

    if np.linalg.det(matrix) == 0:
        return None

    return np.linalg.inv(matrix).tolist()