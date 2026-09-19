def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    m = len(a)
    n = len(a[0])
    ans=[]
    for j in range(n):
        new_row=[]
        for i in range(m):
            new_row.append(a[i][j])
        ans.append(new_row)

    return ans