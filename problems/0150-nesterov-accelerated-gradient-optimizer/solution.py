import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    # Your code here
    #concept Before calculating the gradient, let me first imagine where momentum is going to take me, and calculate the gradient there
    #Normal Momentum:
    # current position → calculate gradient → move

    # Nesterov:
    # current position → look ahead → calculate gradient there → move
    lookahead = parameter - momentum * velocity
    grad = grad_fn(lookahead)

    velocity = momentum * velocity + learning_rate * grad
    parameter = parameter - velocity
    return np.round(parameter, 5), np.round(velocity, 5)