'''
Given N >= 8 point pairs
Estimate fundamental matrix F by solving linear least square
'''

import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def EstimateFundamentalMatrix(x1s, x2s):
    # x2 F x1 = 0
    length = len(x1s)
    A = []

    for i in range(length):
        x1 = x1s[0]
        y1 = x1s[1]
        x2 = x2s[0]
        y2 = x2s[1]
        A.append([x1 * x2, x1 * y2, x1,
                  y1 * x2, y1 * y2, y1,
                  x2, y2, 1])
        
    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    F = VT[-1, :].reshape((3, 3))

    U, S, VT = np.linalg.svd(F)
    S[2, 2] = 0.
    F = np.matmul(U, np.matmul(S, VT))

    return F