'''
Given N >= 8 correspondances between two images
Estimate inlier correspondences using fundamental matrix
'''

from EstimateFundamentalMatrix import *
import random
import cv2

def ransac(image1Coords, image2Coords, MaxIter, threshold):
    NumPairs = len(image1Coords)
    MaxInlier = 0
    MaxInlierList = []
    AllOutlierList = []

    for i in range(MaxIter):
        randPairsIdx = random.sample(range(NumPairs), 16)
        x1s = []
        x2s = []
        for idx in randPairsIdx:
            x1s.append(image1Coords[idx])
            x2s.append(image2Coords[idx])
        x1s = np.array(x1s)
        x2s = np.array(x2s)

        F = EstimateFundamentalMatrix(x1s, x2s)
        InlierList = []
        OutlierList = []
        for j in range(NumPairs):
            x1 = np.array(list(image1Coords[j])+[1])
            x2 = np.array(list(image2Coords[j])+[1])
            x1Fx2 = abs(np.matmul(x2.T, np.matmul(F, x1)))
            if x1Fx2 < threshold:
                InlierList.append(j)
            else:
                OutlierList.append(j)

        if MaxInlier < len(InlierList):
            MaxInlier = len(InlierList)
            MaxInlierList = InlierList
            AllOutlierList = OutlierList

    return MaxInlierList, AllOutlierList

    
def homographyRANSAC(image1Coords, image2Coords, MaxIter, threshold):
    NumPairs = len(image2Coords)
    MaxInlier = 0
    MaxInlierList = []
    AllOutlierList = []

    for i in range(MaxIter):
        randPairsIdx = random.sample(range(NumPairs), 4)
        x1s = []
        x2s = []
        for idx in randPairsIdx:
            x1s.append(image1Coords[idx])
            x2s.append(image2Coords[idx])
        x1s = np.float32(x1s)
        x2s = np.float32(x2s)

        h = cv2.getPerspectiveTransform(x1s,x2s)
        InlierList = []
        OutlierList = []
        for j in range(NumPairs):
            x1 = np.array(list(image1Coords[j])+[1])
            x2 = np.array(list(image2Coords[j])+[1])
            predict = np.matmul(h, np.array([x1[0],x1[1],1]).transpose())
            if np.linalg.norm(x2-predict) < threshold:
                InlierList.append(j)
            else:
                OutlierList.append(j)

        if MaxInlier < len(InlierList):
            MaxInlier = len(InlierList)
            MaxInlierList = InlierList
            AllOutlierList = OutlierList

    return MaxInlierList, AllOutlierList
    

