import math

import numpy as np


def dct(vector):
    # lunghezza del vettore
    l = len(vector)

    # pre-allocazione
    c = np.zeros(l)

    for i in range(l):
        s = 0
        for j in range(l):
            s += vector[j] * np.cos(i * np.pi * ((2 * j + 1) / (2 * l)))
            if i == 0:
                alpha = 1 / math.sqrt(l)
            else:
                alpha = math.sqrt(2 / l)

            c[i] = alpha * s
    return c

def idct(C):
    l = len(C)
    t = np.zeros(l, dtype=np.float)
    for j in range(l):
        s = 0
        for i in range(l):
            if i == 0:
                alpha = 1 / math.sqrt(l)
            else:
                alpha = math.sqrt(2 / l)

            s += C[i] * alpha * np.cos(i * np.pi * ((2 * j + 1) / (2 * l)))
        t[j] = s
    return t


def dct2(Matrix):
    n, m = Matrix.shape
    C = np.zeros((n, m), dtype='float')
    for j in range(m):
        C[:, j] = dct(Matrix[:, j])

    for i in range(n):
        C[i, : ] = dct(C[i, :])

    return C


