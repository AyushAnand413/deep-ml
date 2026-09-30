import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    
    max_score = np.max(scores)
    
    shifted = scores - max_score
    
    return shifted - np.log(np.sum(np.exp(shifted)))