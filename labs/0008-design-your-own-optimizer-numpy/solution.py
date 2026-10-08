
import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter

    # Set AdamW hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.01

    # Initialize optimizer state on the first call
    if 'm' not in state:
        state['m'] = np.zeros_like(param)
        state['v'] = np.zeros_like(param)
        state['step'] = 0

    # Update timestep
    state['step'] += 1
    t = state['step']

    # Update m and v 
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad
    state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)

    # Apply bias correction
    m_hat = state['m'] / (1 - beta1 ** t)
    v_hat = state['v'] / (1 - beta2 ** t)

    # calc adam_update
    adam_update = lr * m_hat /(np.sqrt(v_hat) + epsilon)

    # Calculate decoupled weight decay
    decay = lr * weight_decay * param

    # Update params
    new_param = param - adam_update - decay

    return new_param, state
