import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    X_np  = np.array(X)
    y_np = np.array(y)
    #formula implemantation
    X_trans = X_np.T   # X^T
    XtX = X_trans.dot(X_np) # X^T * X
    XtX_inverse = np.linalg.inv(XtX) # (X^T X)^-1
    Xty = X_trans.dot(y_np)   # X^T y
    theta = XtX_inverse.dot(Xty) # Final θ


	return [round(val, 4) for val in theta]