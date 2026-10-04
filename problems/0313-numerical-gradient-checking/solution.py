import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    #we need to calc manually the gardient and compare to the analytical grad 

    numerical_grad = np.zeros_like(x,dtype=float)

    for i in range(len(x)):
        org = x[i]

        x[i] = org + epsilon
        f_plus = f(x)
        
        x[i] = org - epsilon
        f_minus = f(x)

        x[i] = org
        numerical_grad[i] = (f_plus - f_minus) / (2 * epsilon)


    #now compare
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = (np.linalg.norm(numerical_grad)+
        np.linalg.norm(analytical_grad)
    )

    relative_error = numerator / denominator if denominator != 0 else 0.0

    return numerical_grad, relative_error


