import numpy as np
from mydct import dct


def dct2(Matrix):
    n, m = Matrix.shape
    C = np.zeros((n, m), dtype='float')
    for j in range(m):
        C[:, j] = dct(Matrix[:, j])

    for i in range(n):
        C[i, : ] = dct(C[i, :])

    return C


