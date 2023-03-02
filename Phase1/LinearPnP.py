'''
Given 3D-2D correspondences X - x
and intrinsic matrix K
Estimate camera pose (C, R)
'''
import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def linearPNP(Xs, xs, K):
    length = Xs.shape[0]
    A = np.zeros((2 * length, 12))

    for i in range(length):
        XHomo = np.hstack((Xs[i], np.ones(1)))
        x, y = xs[i]
        A[2*i : 2*i+2, :] = np.block([
            [XHomo, np.zeros((1, 4)), -x * XHomo],
            [np.zeros((1, 4)), XHomo, -y * XHomo]
        ])

    _, sig, VT = np.linalg.svd(A)
    P = VT[np.argmin(sig), :].reshape((3, 4))

    Kinv = np.linalg.inv(K)
    U, sig, VT = np.linalg.svd(Kinv @ P[:, :3])
    R = U @ VT
    C = Kinv @ P[:, 3] / sig[0]
    if np.linalg.det(R) < 0:
        R = -R

    return R, C