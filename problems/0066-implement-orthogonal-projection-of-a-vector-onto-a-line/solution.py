import numpy as np

def orthogonal_projection(v, L):
    v = np.array(v)
    L = np.array(L)

    projection = (np.dot(v, L) / np.dot(L, L)) * L

    return [round(x, 3) for x in projection]