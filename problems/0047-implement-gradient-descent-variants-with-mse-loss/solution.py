import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m = X.shape[0];

    for epoch in range(n_epochs):
        
        if method == 'batch':
            #calc grad over all sample
            pred = X @ weights
            error = pred-y

            dw = (2/m)* (X.T @ error)
            #no bias mention so leave it
            weights -= learning_rate*dw 

        elif method=='stochastic':
            #calc grad over sample

            for i in range(m):
                x_i = X[i]
                y_i = y[i]

                pred = x_i @ weights
                error = pred - y_i

                dw = 2*x_i*error

                weights -= learning_rate*dw 
        
        elif method == "mini_batch":

            for start in range(0, m, batch_size):

                end = min(start + batch_size, m)

                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = X_batch @ weights
                error = predictions - y_batch

                batch_size_actual = len(X_batch)

                dw = (2 / batch_size_actual) * (X_batch.T @ error)

                weights -=learning_rate * dw

    return weights

