'''
Given initialized camera poses and 3D points
Refine all of them
'''
import numpy as np
from scipy.optimize import least_squares
from Reprojection import *
import sys
from scipy.spatial.transform import Rotation

# Don't generate pyc codes
sys.dont_write_bytecode = True

def R2quaternion(R):
    return Rotation.from_matrix(R).as_quat()

def quaternion2R(q):
    return Rotation.from_quat(q).as_matrix()

def flattenCRX(C, R, X):
    return np.hstack((C.reshape(-1), R.reshape(-1), X.reshape(-1)))

def recoverCRX(x0):
    C = x0[:3].reshape(3)
    R = x0[3:12].reshape(3, 3)
    X = x0[12:].reshape(-1, 3)
    return C, R, X

def flattenCqX(C, q, X):
    return np.hstack((C.reshape(-1), q.reshape(-1), X.reshape(-1)))

def recoverCqX(x0):
    C = x0[:3].reshape(3)
    q = x0[3:7]
    X = x0[7:].reshape(-1, 3)
    return C, q, X

def reprojErrorCqX(x0, x, K, V):
    allError = 0
    

    return allError

def BundleAdjustment(Cinit, Rinit, Xinit, x, K, V):
    qinit = R2quaternion(Rinit)
    x0 = flattenCqX(Cinit, qinit, Xinit)
    result = least_squares(reprojErrorCqX, x0, args=(x, K, V), method='lm')
    C, q, X = recoverCqX(result.x)
    R = quaternion2R(q)
    return C, R, X