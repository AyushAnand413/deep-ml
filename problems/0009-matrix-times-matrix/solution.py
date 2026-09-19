def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a) == 0 or len(b) == 0 or len(a[0]) != len(b):
        return -1
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    c = []
    for i in range(m):
        new_row = []
        for j in range(p):
            dot = 0
            for k in range(n):
                dot += a[i][k] * b[k][j]
            new_row.append(dot)
        c.append(new_row)
    return c