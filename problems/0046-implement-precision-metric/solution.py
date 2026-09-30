import numpy as np
def precision(y_true, y_pred):
	# Your code here
	#precision = tp / (tp+fp)

	#tp :- predicted 1 actual also 1
	#fp :- predicted 1 actual 0

	tp = np.sum((y_true==1) & (y_pred==1))
	fp = np.sum((y_true==0) & (y_pred==1))

	if tp+fp == 0 :
		return  0.0
	return tp/(tp+fp)
	
