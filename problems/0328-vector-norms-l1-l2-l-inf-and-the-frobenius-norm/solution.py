import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    ans=0
    if norm_type=='l1' :
        for val in arr.flat :
            ans += abs(val)
        return float(ans) 

    elif norm_type=='l2' :
        for val in arr.flat:
            ans+= val*val;
        return float(np.sqrt(ans))

    elif norm_type=='linf' :
        for val in arr.flat:
            ans = max(ans,abs(val))
        return float(ans)

    elif norm_type=='frobenius' :
        if arr.ndim !=2:
            raise  ValueError("not 2d array")
        for val in arr.flat:
            ans+= val*val;
        return float(np.sqrt(ans))
    else :
        raise ValueError("invlaid norm")
            


