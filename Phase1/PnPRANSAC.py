from Reprojection import *
from LinearPnP import *
import sys
import random

# Don't generate pyc codes
sys.dont_write_bytecode = True

def reprojsErrorSquare(x, xProj):
    return (x[0] - xProj[0])**2 + (x[1] - xProj[1])**2

def PnPRANSAC(Xs, xs, K, MaxIter, threshold):
    length = Xs.shape[0]
    MaxInlier = 0
    Rbest = 0
    Cbest = 0

    for _ in range(MaxIter):
        randIdx = random.sample(range(length), 6)
        Xselected = np.array([Xs[i] for i in randIdx])
        xselected = np.array([xs[i] for i in randIdx])
        R, C = linearPNP(Xselected, xselected, K)

        inliers = []
        for i in range(length):
            xProj = reprojection(K, R, C, [Xs[i]])[0]
            error = reprojsErrorSquare(xs[i], xProj)
            if error < threshold:
                inliers.append(i)

        if MaxInlier < len(inliers):
            MaxInlier = len(inliers)
            Rbest = R
            Cbest = C

    return Rbest, Cbest