import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        G: Accumulated squared gradients
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_G)
    """
    # Your code here
    #adgrad main idea is that it remember all previous squared grads and use that memory to control the step size
    #and see it rembers every grads but in rmsprop it only remembers recent ones
    G = G + grad**2
    parameter -= learning_rate*grad/(np.sqrt(G)+epsilon)

    #see the term root g which is didved by lr so when we accumulate grads and squre it root g also grows and hence dec lr 
    return np.round(parameter, 5), np.round(G, 5)