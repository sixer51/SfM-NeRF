'''
Given fundamental matrix F
Estimate essential matrix E
'''

import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def EssentialFromFundamental(F, K):
    F = np.array(F)
    K = np.array(K)

    E = np.matmul(K.T, np.matmul(F, K))
    U, S, VT = np.linalg.svd(E)
    # print(S)
    S = np.eye(3, dtype=float)
    S[2, 2] = 0.
    E = np.matmul(U, np.matmul(S, VT))

    return E