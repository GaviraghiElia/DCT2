import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dctn, dct
import seaborn as sns

import mydct
import mydct2

def test_dct():
    vector = np.array([231, 32, 233, 161, 24, 71, 140, 245])

    print("test open dct")
    print(dct(vector, norm='ortho'))

    print("\n test my dct")
    print(mydct.dct(vector))

def test_dctn():
    mat = np.array([
        [231, 32, 233, 161, 24, 71, 140, 245],
        [247, 40, 248, 245, 124, 204, 36, 107],
        [234, 202, 245, 167, 9, 217, 239, 173],
        [193, 190, 100, 167, 43, 180, 8, 70],
        [11, 24, 210, 177, 81, 243, 8, 112],
        [97, 195, 203, 47, 125, 114, 165, 181],
        [193, 70, 174, 167, 41, 30, 127, 245],
        [87, 149, 57, 192, 65, 129, 178, 228],
    ])
    print("\n\n test open dctn2")
    print(dctn(mat, norm='ortho'))

    print("test my dct2")
    print(mydct2.dct2(mat))


def performance_test(start, nmatrix):
    M = []
    i = 1
    n = start
    while i <= nmatrix:
        i += 1
        problem = np.random.randint(0, 255, size=(n, n))
        tic = time.perf_counter()
        dctn(problem, norm='ortho')
        toc = time.perf_counter()

        M.append({
            'size': n,
            'time': float(toc - tic),
            'type': 'library'
        })

        M.append({
            'size': n,
            'time': pow(i, 2),
            'type': 'n^2'
        })

        w = pow(i, 2) * math.log(i, 2)

        M.append({
            'size': n,
            'time': w,
            'type': 'n^2 log n'
        })

        print('[OPEN] Matrix ({0},{0}) took {1:.4f}ms to complete.'.format(n, float(toc-tic)))

        #tic = time.perf_counter()
        #mydct2.dct2(problem)
        #toc = time.perf_counter()

        #M.append({
        #    'size': n,
        #    'time': float(toc - tic),
        #    'type': 'custom'
        #})

        #print('[CUSTOM] Matrix ({0},{0}) took {1:.4f}ms to complete.'.format(n, float(toc-tic)))
        #print('\n')

        n = n * 2

    return M

def plot_results(results):
    df = pd.DataFrame.from_dict(results)
    ax = plt.axes()
    sns.lineplot(data=df, x='size', y='time', hue='type', marker='o', ax=ax)
    ax.set_yscale('log')
    plt.show()


# Plot style
sns.set_style('whitegrid')
test_dct()
test_dctn()
results = performance_test(400, 6)
plot_results(results)