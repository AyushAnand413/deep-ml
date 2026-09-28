import numpy as np

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    mse = np.mean((y_true - y_pred) ** 2)

    rmse_res = np.sqrt(mse)

    return round(rmse_res, 3)