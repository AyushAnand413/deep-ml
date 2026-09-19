def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	m = len(matrix)
	n = len(matrix[0])
	ans=[]
	for i in range(m):
		new_row=[]
		for j in range(n):
			new_row.append(matrix[i][j]*scalar)
		ans.append(new_row)
	return ans