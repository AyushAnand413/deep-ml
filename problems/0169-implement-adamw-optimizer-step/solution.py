import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here

    #so adamW is just adding weight_decay in adam we do this to reduce the weight slighty bcz very large weights sometimes overfit the model so by this we improve genralization

    #update m and v
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)

    #apply bias correction
    m_hat = m /(1-beta1**t)
    v_hat = v /(1-beta2**t)

    #adam update 
    adam_update = lr * m_hat /(np.sqrt(v_hat)+epsilon)

    #calc decoupled weight decay 
    decay = lr*weight_decay*w

    #update params
    w_new = w - adam_update -decay
    return w_new,m,v
