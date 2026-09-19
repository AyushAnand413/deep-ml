import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	#lets calc magnitude of both v1 and v2 
	mag_v1=0
	mag_v2=0
	for val in v1.flat :
		mag_v1 += val*val
	mag_v1 = np.sqrt(mag_v1)
	for val in v2.flat :
		mag_v2 += val*val
	mag_v2 = np.sqrt(mag_v2)
	#now dot prod
	ans=0
	for i in range(len(v1)):
		ans+= v1[i]*v2[i]

	result = float(ans/(mag_v1*mag_v2))
	return result




