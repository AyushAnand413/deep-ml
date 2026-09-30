import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    
    #imputation 
    #calc mean column wise for each feature as we 
    #replace nan with thier feature means 

    column_means = np.nanmean(X_clean, axis=0)
    

    for j in range(X_clean.shape[1]):
        missing = np.isnan(X_clean[:, j])
        X_clean[missing, j] = column_means[j]

    
    return X_clean
