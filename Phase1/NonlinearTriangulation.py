'''
Given two camera poses and linearly triangulated points X
Refine 3D points by minimizing reprojection error
'''

import numpy as np

def reprojection(K, R, C, Xs):
    t = C.reshape((3,1))
    T = np.hstack((R, t))
    P = np.matmul(K, T)

    reprojImagePoints = []
    for X in Xs:
        X = np.hstack((X, np.ones(1)))
        x = P @ X
        u = x[0] / x[2]
        v = x[1] / x[2]
        reprojImagePoints.append((u, v))

    return reprojImagePoints
