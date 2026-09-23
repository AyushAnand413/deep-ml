import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sigmoid = 1/(1+np.exp(-x))

	return{
		"sigmoid" : sigmoid*(1-sigmoid),
		"tanh" : 1 - np.tanh(x)**2,
		"relu" : 0.0 if x <= 0 else 1.0
	}