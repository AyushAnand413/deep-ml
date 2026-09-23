import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	mse_values = []
	weights = initial_weights.astype(float).copy()
	bias = float(initial_bias)

	for _ in range(epochs):

		#forward pass
		z = features @ weights + bias
		predictions = 1 / (1+np.exp(-z)) #sigmoid

		#mse before updation of weight 
		error = predictions - labels
		mse = np.mean(error**2)
		mse_values.append(round(mse, 4))

		#backprop
		#sigmoid derivative
		sigmoid_derivative = predictions * (1-predictions)

		dz = 2*error*sigmoid_derivative

		dw = np.mean(features * dz[:,np.newaxis],axis=0)
		db = np.mean(dz)

		#gradient descent
		weights -= learning_rate*dw
		bias -= learning_rate*db
	
	return weights, bias , mse_values