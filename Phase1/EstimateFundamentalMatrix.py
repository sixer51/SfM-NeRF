'''
Given N >= 8 point pairs
Estimate fundamental matrix F by solving linear least square
'''

import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def EstimateFundamentalMatrix(x1s, x2s):
    # x1^T F x2 = 0
    length = len(x1s)
    A = []

    for i in range(length):
        x1 = x1s[i, 0]
        y1 = x1s[i, 1]
        x2 = x2s[i, 0]
        y2 = x2s[i, 1]
        A.append([x1 * x2, y1 * x2, x2,
                  x1 * y2, y1 * y2, y2,
                  x1, y1, 1])
        
    A = np.array(A)
    _, sig, VT = np.linalg.svd(A)
    F = VT[np.argmin(sig), :].reshape((3, 3))

    U, S, VT = np.linalg.svd(F)

    S[2] = 0.
    F = np.matmul(U, np.matmul(np.diag(S), VT))

    return F