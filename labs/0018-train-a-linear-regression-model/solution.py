import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    learning_rate = 0.01
    epoch = 1000
    for _ in range(epoch):

        # pred
        prediction = X @ W + b 

        #error 
        error = prediction - y 
        #mse
        mse = np.mean(error ** 2)

        #gardients
        dw =  (2/len(y)) * (X.T @ error)
        db  = (2/len(y)) * np.sum(error)
        
        # update weights
        W -= learning_rate*dw
        b -= learning_rate*db 

    return W,b
