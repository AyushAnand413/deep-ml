import math

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients by global norm.

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed global norm

    Returns:
        List of clipped gradient arrays
    """
    # Your code here
    #l2 norm 
    #square all gards and sum it 
    # take square root 
    #then we check this global norm to the given max norm if its greater we apply scaling factor which is max/global and multiply all grads with this and return it  

    total = 0

    for grads in gradients:
        for g in grads:
            total += g ** 2

    global_norm = math.sqrt(total)

    if global_norm > max_norm:
        scale = max_norm / global_norm

        gradients = [
            [g * scale for g in grads]
            for grads in gradients
        ]

    return gradients