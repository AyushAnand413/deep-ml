import numpy as np

def compute_chain_rule_gradient(functions, x):
    value = x
    derivative = 1.0

    for func in reversed(functions):
        if func == "square":
            derivative *= 2 * value
            value = value ** 2

        elif func == "sin":
            derivative *= np.cos(value)
            value = np.sin(value)

        elif func == "exp":
            derivative *= np.exp(value)
            value = np.exp(value)

        elif func == "log":
            derivative *= 1 / value
            value = np.log(value)

    return derivative