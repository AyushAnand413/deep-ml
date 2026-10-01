import math
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	max_x = max(x)

	#calc softmax 
	exp_x = [math.exp(value-max_x) for value in x]
	total = sum(exp_x)


	s = [value / total for value in exp_x] #softmax prob
	#jacobian matrix 
	n = len(x)
	#create empty mat first 
	J = [[0.0 for _ in range(n)] for _ in range(n)]	

	for i in range(n):
		for j in range(n):
			if i==j:
				J[i][j] = s[i] * (1 - s[i])
			else:
				J[i][j] = -s[i] * s[j]
	
	return J

