def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	"""
	Perform RMSProp optimization update.
	
	Args:
		params: List of parameter values
		grads: List of gradients for each parameter
		cache: List of cache values (moving average of squared gradients)
		lr: Learning rate
		beta: Decay rate for moving average
		epsilon: Small constant for numerical stability
	
	Returns:
		Tuple of (updated_params, updated_cache)
	"""
	# Your code here
	updated_param =[]
	updated_cache =[]

	for p,g,c in zip(params,grads,cache):
		squared_gradient = g**2 # we sq the  grad to make all grads comparable
		
		new_cache = beta*c + (1-beta)*squared_gradient # we upadte the moving avg of squared_gradient
		
		update = lr * g /((new_cache**0.5)+epsilon) #rmsprop 
		new_param = p - update # update parameter in opp dir of grads

		updated_param.append(new_param)
		updated_cache.append(new_cache)

	return updated_param,updated_cache

