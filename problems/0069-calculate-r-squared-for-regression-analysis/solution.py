
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	mean_y = np.mean(y_true)

	sst= np.sum((y_true - mean_y)**2)
	
	sse = np.sum((y_true - y_pred)**2)

	return 1 - (sse/sst)