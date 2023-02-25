'''
Given two camera poses and linearly triangulated points X
Refine 3D points by minimizing reprojection error
'''

import numpy as np
from scipy.optimize import least_squares

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
        reprojImagePoints.append([u, v])

    return np.array(reprojImagePoints)

def flattenWorldPoint(Xs):
    return Xs.reshape(-1)

def recoverWorldPoints(x0):
    return x0.reshape(-1, 3)

def reprojError(x0, K, R1, C1, R2, C2, x1s, x2s):
    reprojImagePoints1 = reprojection(K, R1, C1, recoverWorldPoints(x0))
    reprojImagePointsFlatten1 = reprojImagePoints1.reshape(-1)
    x1sError = x1s.reshape(-1) - reprojImagePointsFlatten1

    reprojImagePoints2 = reprojection(K, R2, C2, recoverWorldPoints(x0))
    reprojImagePointsFlatten2 = reprojImagePoints2.reshape(-1)
    x2sError = x2s.reshape(-1) - reprojImagePointsFlatten2

    error = np.hstack((x1sError, x2sError))
    return error

def nonlinearTriangulation(K, R1, C1, R2, C2, x1s, x2s, Xs):
    x0 = flattenWorldPoint(Xs)
    result = least_squares(reprojError, x0, args=(K, R1, C1, R2, C2, x1s, x2s), method='lm')
    Xs = recoverWorldPoints(result.x)
    return Xs

