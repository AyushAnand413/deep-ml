import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	correect = np.sum(y_true==y_pred)
	total = len(y_true)

	return correect/total