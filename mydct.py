import math
import numpy as np

def dct(vector):
    # lunghezza del vettore
    N = len(vector)

    # pre-allocazione per ottimizzare il codice
    c = np.zeros(N)

    # ottimizzazione del codice calcolando alpha una sola volta
    alphaZero = 1 / math.sqrt(N)
    alphaNotZero = math.sqrt(2 / N)

    for k in range(N):
        if k == 0:
            alpha = alphaZero
        else:
            alpha = alphaNotZero
        s = 0
        for i in range(N):
            s += vector[i] * np.cos(np.pi * k * ((2 * i + 1) / (2 * N)))
            c[k] = alpha * s

    return c
