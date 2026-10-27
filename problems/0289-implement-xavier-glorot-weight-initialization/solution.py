
import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    """
    Initialize a weight matrix using Xavier/Glorot initialization.

    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions.
        mode: 'uniform' or 'normal' initialization.
        seed: Random seed for reproducibility (optional).

    Returns:
        Initialized weight matrix as a NumPy array.

    Raises:
        ValueError: If mode is neither 'uniform' nor 'normal'.
    """
    ## We use Xavier initialization to prevent weights from being too large or too small during training and suited for tanh and sigmoid 
    if seed is not None:
        np.random.seed(seed)

    fan_in, fan_out = shape

    if mode == 'uniform':
        limit = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, shape)
    elif mode == 'normal':
        sigma = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, sigma, shape)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")

    return weights
