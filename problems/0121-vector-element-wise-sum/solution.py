def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.

	#we compare len
	if(len(a) !=len(b)) :
		return -1;

	ans=[]
	
	for i in range(len(a)):
		ans.append(a[i]+b[i])
	return ans

