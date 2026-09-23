import numpy as np

def jacobian_matrix(f, x, h=1e-5):
    x = np.array(x, dtype=float)

    f0 = np.array(f(x))

    m = len(f0)      # number of outputs
    n = len(x)       # number of inputs

    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += h
        x_minus[j] -= h

        f_plus = np.array(f(x_plus))
        f_minus = np.array(f(x_minus))

        J[:, j] = (f_plus - f_minus) / (2 * h)

    return J.tolist()