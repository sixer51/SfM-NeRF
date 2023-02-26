import glob
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

class FeaturesAssociationMap:
    def __init__(self, matchFileDirPath):
        self.visibilityMap = None
        self.featuresColor = None
        self.featuresU = None
        self.featuresV = None
        self._load(matchFileDirPath)

    def _load(self,matchFileDirPath):
        files = sorted(glob.glob(matchFileDirPath + 'matching*.txt'))
        # print(files)
        fileNum = len(files)
        imageNum = fileNum + 1
        
        featuresU = []
        featuresV = []
        featuresColor = []
        visMap = []
        
        for matchFileIdx, matchFileName in enumerate(files):
            with open(matchFileName) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    lineElements = line.split(' ')[:-1]
                    # u = np.zeros((1,imageNum))
                    # v = np.zeros((1,imageNum))
                    # featureVis = np.zeros((1,imageNum))
                    u = np.zeros(imageNum)
                    v = np.zeros(imageNum)
                    featureVis = np.zeros(imageNum,dtype=bool)
                    color = np.int16([int(lineElements[1]),int(lineElements[2]),int(lineElements[3])])
                    
                    featureVis[matchFileIdx] = 1
                    u[matchFileIdx] = float(lineElements[4])
                    v[matchFileIdx] = float(lineElements[5])

                    lineElements = lineElements[6:]
                    for i in range(len(lineElements)//3):
                        image2Id = int(lineElements[0]) - 1
                        featureVis[image2Id] = 1
                        u[image2Id] = float(lineElements[1])
                        v[image2Id] = float(lineElements[2])
                        lineElements = lineElements[3:]
                    featuresU.append(u)
                    featuresV.append(v)
                    visMap.append(featureVis)
                    featuresColor.append(color)
        
        self.featuresU = np.vstack(featuresU)
        self.featuresV = np.vstack(featuresV)
        self.featuresColor = np.vstack(featuresColor)
        self.visibilityMap = np.vstack(visMap)
    
    def get_feature_matches(self, imagePair):
        image1Id,image2Id = imagePair
        image1Id -= 1
        image2Id -= 1

        idxs = np.where(np.logical_and(self.visibilityMap[:,image1Id], self.visibilityMap[:,image2Id]))[0]

        image1Coords = [self.featuresU[idxs,image1Id], self.featuresV[idxs,image1Id]]
        image2Coords = [self.featuresU[idxs,image2Id], self.featuresV[idxs,image2Id]]

        image1Coords = np.vstack(image1Coords).T
        image2Coords = np.vstack(image2Coords).T

        return image1Coords, image2Coords, idxs

def filterCoordsByIdxs(coords1, coords2, featuresIdxs, idxs):
    coords1_filtered = coords1[idxs]
    coords2_filtered = coords2[idxs]
    featuresIdxs_filtered = featuresIdxs[idxs]
    return coords1_filtered, coords2_filtered, featuresIdxs_filtered



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
    