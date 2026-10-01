import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	probabilities = []

	for x in features:
		z = sum(x[i]*weights[i] for i in range(len(weights)))+bias
		probab = 1 / (1+math.exp(-z))
		probabilities.append(probab)

	mse = sum((probabilities[i] - labels[i]) ** 2 for i in range(len(labels))) / len(labels)

	probabilities = [round(p, 4) for p in probabilities]
	mse = round(mse, 4)
	return probabilities, mse