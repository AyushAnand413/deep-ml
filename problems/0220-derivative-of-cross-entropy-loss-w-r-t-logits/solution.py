import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	#we calc softmax first then one hot we write and then gradient
	logits= np.array(logits, dtype=float)
	exp_logits = np.exp(logits - np.max(logits))

	q = exp_logits / np.sum(exp_logits)

	y= np.zeros(len(logits))
	y[target]=1

	return (q-y).tolist()



