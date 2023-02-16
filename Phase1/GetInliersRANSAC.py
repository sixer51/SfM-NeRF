'''
Given N >= 8 correspondances between two images
Estimate inlier correspondences using fundamental matrix
'''

from EstimateFundamentalMatrix import *
import random

def ransac(MatchPairs, MaxIter, threshold):
    NumPairs = len(MatchPairs)
    MaxInlier = 0
    MaxInlierList = []
    AllOutlierList = []

    for i in range(MaxIter):
        randPairsIdx = random.sample(range(NumPairs), 8)
        x1s = []
        x2s = []
        for idx in randPairsIdx:
            x1s.append(MatchPairs[idx].coords1)
            x2s.append(MatchPairs[idx].coords2)
        x1s = np.array(x1s)
        x2s = np.array(x2s)

        F = EstimateFundamentalMatrix(x1s, x2s)
        InlierList = []
        OutlierList = []
        for j in range(NumPairs):
            x1 = np.array(list(MatchPairs[j].coords1)+[1])
            x2 = np.array(list(MatchPairs[j].coords2)+[1])
            x2Fx1 = abs(np.matmul(x2.T, np.matmul(F, x1)))
            if x2Fx1 < threshold:
                InlierList.append(j)
            else:
                OutlierList.append(j)

        if MaxInlier < len(InlierList):
            MaxInlier = len(InlierList)
            MaxInlierList = InlierList
            AllOutlierList = OutlierList

    return MaxInlierList, AllOutlierList

    

    

