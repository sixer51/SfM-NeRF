'''
Given effective matrix E
Estimate camera pose
'''
import numpy as np
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

def extractCameraPose(E):
    U, D, VT = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R = [np.matmul(U,np.matmul(W,VT)),
         np.matmul(U,np.matmul(W,VT)),
         np.matmul(U,np.matmul(W.T,VT)),
         np.matmul(U,np.matmul(W.T,VT))]
    
    T = [U[:,2],
         -U[:,2],
         U[:,2],
         -U[:,2]]
    
    for i in range(4):
        if np.linalg.det(R[i])<0:
            R[i] = -R[i]
            T[i] = -T[i]

    return R,T

    

