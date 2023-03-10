'''
Given two camera poses and correspondences (point pairs)
Triangulate 3D points
'''

import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def imagePoint2crossMatrix(x):
    u = x[0]
    v = x[1]
    return np.array([[0, -1, v],
                     [1, 0, -u],
                     [-v, u, 0]])

def EstimateWorldPoint(K, xs, Rs, Ts):
    A = np.zeros([3 * len(xs), 4])
    for i in range(len(xs)):
        crossx = imagePoint2crossMatrix(xs[i])
        t = Ts[i].reshape((3,1))
        T = np.hstack((Rs[i], t))
        P = np.matmul(K, T)
        A[3*i:3*i+3, :] = np.matmul(crossx, P)
    
    A = np.array(A)
    _, sigma, VT = np.linalg.svd(A)
    X = VT[np.argmin(sigma), :]
    X = X/X[-1]
    return X[:3]

def LinearTriangulation(K, x1s, x2s, R1, T1, R2, T2):
    length = len(x1s)
    Xs = []
    for i in range(length):
        X = EstimateWorldPoint(K, [x1s[i], x2s[i]], [R1, R2], [T1, T2])
        Xs.append(X)
        
    return np.array(Xs)
