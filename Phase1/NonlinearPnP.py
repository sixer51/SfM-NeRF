'''
Given N >= 6 3D-2D correspondences X -x
and linearly estimated camera pose (C, R)
Refine camera pose
'''

import numpy as np
from scipy.optimize import least_squares
from Reprojection import *
import sys
from scipy.spatial.transform import Rotation

# Don't generate pyc codes
sys.dont_write_bytecode = True

def flattenCR(C, R):
    return np.hstack((C.reshape(-1), R.reshape(-1)))

def recoverCR(x0):
    C = x0[:3].reshape(3)
    R = x0[3:].reshape(3, 3)
    return C, R

def R2quaternion(R):
    return Rotation.from_matrix(R).as_quat()

def quaternion2R(q):
    return Rotation.from_quat(q).as_matrix()

def flattenCq(C, q):
    return np.hstack((C.reshape(-1), q.reshape(-1)))

def recoverCq(x0):
    C = x0[:3].reshape(3)
    q = x0[3:]
    return C, q

def reprojErrorCR(x0, K, xs, Xs):
    C, R = recoverCR(x0)
    reprojImagePoints = reprojection(K, R, C, Xs)
    error = xs.reshape(-1) - reprojImagePoints.reshape(-1)
    return error

def reprojErrorCq(x0, K, xs, Xs):
    C, q = recoverCq(x0)
    R = quaternion2R(q)
    x0CR = flattenCR(C, R)
    return reprojErrorCR(x0CR, K, xs, Xs)

def nonlinearPnPCR(K, xs, Xs, Cinit, Rinit):
    x0 = flattenCR(Cinit, Rinit)
    result = least_squares(reprojErrorCR, x0, args=(K, xs, Xs), method='lm')
    C, R = recoverCR(result.x)
    return C, R

def nonlinearPnPCq(K, xs, Xs, Cinit, Rinit):
    qinit = R2quaternion(Rinit)
    x0 = flattenCq(Cinit, qinit)
    result = least_squares(reprojErrorCq, x0, args=(K, xs, Xs), method='lm')
    C, q = recoverCq(result.x)
    R = quaternion2R(q)
    return C, R
