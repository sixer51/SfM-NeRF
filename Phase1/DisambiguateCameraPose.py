'''
Given four camera pose configurations
and triangulated points
Find the unnique camera pose
'''
import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def countPointsSatisfyCheiralityCondition(R, T, Xs):
    satisfiedCount = 0
    for X in Xs:
        if np.matmul(R[2,:],np.subtract(X,T))>0:
            satisfiedCount += 1
    return satisfiedCount

def disambiguateCameraPose(Rset, Tset, Xset):
    mostSatisfiedCount = 0
    bestR = None
    bestT = None
    bestX = None

    for i in range(4):
        currentSatisfiedCount = countPointsSatisfyCheiralityCondition(Rset[i], Tset[i], Xset[i])
        if currentSatisfiedCount > mostSatisfiedCount:
            mostSatisfiedCount = currentSatisfiedCount
            bestR = Rset[i]
            bestT = Tset[i]
            bestX = Xset[i]

    
    return bestR, bestT, bestX