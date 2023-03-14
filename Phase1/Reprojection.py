import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def reprojection(K, R, C, Xs):
    t = C.reshape((3,1))
    T = np.hstack((R, t))
    P = np.matmul(K, T)

    # reprojImagePoints = []
    Xhomo = np.concatenate((Xs, np.ones((Xs.shape[0], 1))), 1)
    x = (P @ Xhomo.T).T
    x[:, 0] = x[:, 0] / x[:, 2]
    x[:, 1] = x[:, 1] / x[:, 2]
    reprojImagePoints = x[:, :2]
    return np.array(reprojImagePoints)