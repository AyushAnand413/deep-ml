import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

	# Standardization: (x - mean) / standard deviation
	mean = np.mean(data, axis=0)
	standard_devaiation = np.std(data, axis=0)
	standardized_data = (data - mean) / standard_devaiation

	# Min-Max normalization: (x - min) / (max - min)
	min_val = np.min(data, axis=0)
	max_val = np.max(data, axis=0)
	normalized_data = (data - min_val) / (max_val - min_val)

	return standardized_data, normalized_data