import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	n_samples, n_features = X.shape
	#we need weights = n_features 
	weights = np.zeros(n_features)
	bias = 0.0
	#weights=[0,0] , bias=0

	losses =[]

	for i in range(iterations):

		z = X @ weights + bias
		pred = 1 / (1 + np.exp(-z))

		#bce loss 
		loss = -np.sum(y * np.log(pred) + 
			(1-y)*np.log(1-pred)
		)

		losses.append(round(loss, 4))

		error = pred - y 
		dw = X.T @ error
		db = np.sum(error)

		weights -= learning_rate * dw
		bias  -= learning_rate * db

	coeff = [bias] + weights.tolist()
	coeff = [round(x, 4) for x in coeff]

	return coeff,losses




