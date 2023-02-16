import glob
import os
import cv2
import sys
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def loadImages(imageDirPath):
    images = []
    for filename in sorted(glob.glob(imageDirPath + '*.png')):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

class MatchPair:
    def __init__(self, RGB_values, image1Coords, image2Coords):
        self.RGB = RGB_values
        self.coords1 = image1Coords
        self.coords2 = image2Coords
    
    def __str__(self):
        return str(self.RGB) + str(self.coords1) + str(self.coords2)
    

class ImageFeatureMatches:
    def __init__(self, image1ID, image2ID):
        self.image1ID = image1ID
        self.image2ID = image2ID
        self.matchPairs = []
    def addMatchPair(self, MatchPair):
        self.matchPairs.append(MatchPair)

    def removeMatchPair(self, idx):
        self.matchPairs.pop(idx)

def getCameraParams(calibResultDirPath):
    calibResultFilePath = calibResultDirPath + "calibration.txt"
    K = np.loadtxt(calibResultFilePath, delimiter=" ")
    return K

def parseMatchFiles(matchFileDirPath):
    featureMatchesMatrix = []
    files = sorted(glob.glob(matchFileDirPath + 'matching*.txt'))
    # print(files)
    fileNum = len(files)
    imageNum = fileNum + 1
    for i in range(imageNum):
        l = []
        for j in range(imageNum):
            matchList = ImageFeatureMatches(i, j)
            l.append(matchList)
        featureMatchesMatrix.append(l)
    
    for matchFileIdx, matchFileName in enumerate(files):
        with open(matchFileName) as f:
            lines = f.readlines()[1:]
            for line in lines:
                lineElements = line.split(' ')[:-1]
                rgb = (int(lineElements[1]),int(lineElements[2]),int(lineElements[3]))
                coords1 = (float(lineElements[4]),float(lineElements[5]))
                lineElements = lineElements[6:]
                for i in range(len(lineElements)//3):
                    image2Id = int(lineElements[0]) - 1
                    coords2 = (float(lineElements[1]),float(lineElements[2]))
                    matchPair = MatchPair(rgb, coords1, coords2)
                    featureMatchesMatrix[matchFileIdx][image2Id].addMatchPair(matchPair)
                    lineElements = lineElements[3:]

    featureMatchesList = []
    for i in range(len(featureMatchesMatrix)):
        for j in range(len(featureMatchesMatrix[0])):
            if(featureMatchesMatrix[i][j].matchPairs):
                featureMatchesList.append(featureMatchesMatrix[i][j])
    
    return featureMatchesList
    