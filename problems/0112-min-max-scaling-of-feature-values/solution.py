import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    vals = np.array(x)
    min_val = np.min(x)
    max_val = np.max(x)

    data = ( vals - min_val) / (max_val - min_val)

    return data.tolist()