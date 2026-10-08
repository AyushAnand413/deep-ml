import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    
    # SGD → uses gradient
    # Adagrad → remembers accumulated squared gradients
    # RMSProp → remembers moving average of squared gradients
    
    #adam combines momentum+rmsprop
    
    #so adam remember grad direction and grad magnitude
    #so we will calc m = moving avg of grads see m will tell us in which dir we are + or - 
    #we then calc v = moving avg of squared grads which tell us the magnitude 
    x = x0.copy()
    m =  np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(num_iterations):
        g = grad(x)
        #update m 
        m_new = beta1*m + (1-beta1)*g
        #update v
        v_new = beta2*v + (1-beta2)*(g**2)

        # Correct the bias caused by initializing m and v with zeros
        #corrected m 
        m_corrected = m_new/(1-beta1**(i+1))

        #corrected v 
        v_corrected = v_new/(1-beta2**(i+1))

        #update parameters
        x -= learning_rate*m_corrected/(np.sqrt(v_corrected)+epsilon)

        m = m_new
        v = v_new

    return x




