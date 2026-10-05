import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    #classification loss using numpy 

    #validate shapes
    assert preds.ndim==2
    N,C = preds.shape
    assert target.shape==(N,)
    assert np.issubdtype(target.dtype, np.integer)
    assert np.all(target >= 0) and np.all(target < C)

    #one hot encoding of target
    y = np.zeros_like(preds)
    y[np.arange(N),target] = 1.0

    #ensure numerical stability
    eps = 1e-12
    p = np.clip(preds, eps, 1.0)

    #per sample  cross entropy loss 
    per_sample = -np.sum(y * np.log(p), axis=1)

    #grads wrt prob 
    d_preds = -y/p 

    #reduction 
    if reduction=='mean':
        loss = float(per_sample.mean())
        d_preds = d_preds/N

    elif reduction=='sum':
        loss = float(per_sample.sum())

    elif reduction=='none':
        loss = per_sample
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


    return loss, d_preds