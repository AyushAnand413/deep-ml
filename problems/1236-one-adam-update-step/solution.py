
import torch

def adam_step(w, grad, m, v, t, lr, beta1, beta2, eps):
    """One Adam update with bias-corrected moments.

    Args:
        w: current parameters (torch.Tensor)
        grad: gradient of the loss w.r.t. w (torch.Tensor)
        m: first moment estimate (torch.Tensor)
        v: second moment estimate (torch.Tensor)
        t: timestep, 1-indexed (int)
        lr: learning rate (float)
        beta1: exp. decay for first moment (float)
        beta2: exp. decay for second moment (float)
        eps: numerical stability constant (float)

    Returns:
        Tuple (w_new, m_new, v_new) as torch.Tensor values.
    """
    # Update moving average of gradients
    m_new = beta1 * m + (1 - beta1) * grad

    # Update moving average of squared gradients
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Apply bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Update parameters using Adam
    w_new = w - lr * m_hat / (torch.sqrt(v_hat) + eps)

    # Return updated parameters and moments
    return w_new, m_new, v_new
