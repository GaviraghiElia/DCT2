import numpy as np
from mydct import dct


def dct2(Matrix):
    n, m = Matrix.shape
    C = np.zeros((n, m), dtype='float')
    for k in range(m):
        C[:, k] = dct(Matrix[:, k])

    for l in range(n):
        C[l, :] = dct(C[l, :])

    return C
