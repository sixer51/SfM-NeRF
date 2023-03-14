from DataLoader import *
from Visualization import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from GetInliersRANSAC import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *
import os

# Don't generate pyc codes
sys.dont_write_bytecode = True

dataDirPath = os.getcwd() + '/Phase1/P3Data/'
images = loadImages(dataDirPath)
K = getCameraParams(dataDirPath)
print(K)

featuresMap = FeaturesAssociationMap(dataDirPath)

imagePairs = [[0, 1], [0, 2], [0, 3], [0, 4]]
allImagePointIdx = set()
inlierPointPairs = []
allC = np.zeros((len(imagePairs)+1, 3))
allR = np.zeros((len(imagePairs)+1, 3, 3))
allIntersectPoints = []
allRefinedX = []

for i in range(len(imagePairs)):
    image1Id = imagePairs[i][0]
    image2Id = imagePairs[i][1]
    image1Coords, image2Coords, featureIdxs = featuresMap.get_feature_matches((image1Id,image2Id))
    InlierIdxs, OutlierIdxs = homographyRANSAC(image1Coords, image2Coords, 2000, 50)
    image1Inliers_homography, image2Inliers_homography, featureIdxs_homography = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, InlierIdxs)
    image1Outliers_homography, image2Outliers_homography, _ = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, OutlierIdxs)
    InlierIdxs, OutlierIdxs = ransac(image1Inliers_homography, image2Inliers_homography, 5000, 0.01)
    
    image1Inliers, image2Inliers, InlierGlobalIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, InlierIdxs)
    image1Outliers, image2Outliers, OutlierGlobalIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, OutlierIdxs)
    featuresMap.removeMatches(OutlierGlobalIdxs, image2Id)
    inlierPointPairs.append(InlierGlobalIdxs)

    matchImage = drawMatchs(images[image1Id], images[image2Id], image1Inliers, image2Inliers, (0,255,0))
    plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.imshow(matchImage)
    plt.show()

image1Inliers = featuresMap.getImagePoints(0, inlierPointPairs[0])
image2Inliers = featuresMap.getImagePoints(1, inlierPointPairs[0])
allImagePointIdx = set(inlierPointPairs[0])
Fest = EstimateFundamentalMatrix(image1Inliers, image2Inliers)
Eest = EssentialFromFundamental(Fest, K)
Rest, Test = extractCameraPose(Eest)

Xest = []
for i in range(len(Rest)):
    X = LinearTriangulation(K, image1Inliers, image2Inliers, np.eye(3), np.zeros((3,1)), Rest[i], Test[i])
    Xest.append(X)

drawWorldPoints(Xest)
Rest, Test, Xest = disambiguateCameraPose(Rest, Test, Xest)
allC[0] = Test
allR[0] = Rest
drawWorldPoints([Xest])

XestNonlinear = nonlinearTriangulation(K, np.eye(3), np.zeros((3,1)), Rest, Test, image1Inliers, image2Inliers, Xest)
featuresMap.updateWorldPoints(list(allImagePointIdx), XestNonlinear)
drawWorldPoints([XestNonlinear])
allRefinedX.append(XestNonlinear)

# start including image 3, 4, 5
for imagePairID in range(1, 4):
    imagePair = imagePairs[imagePairID]
    imageID = imagePair[1]
    image1Inliers = featuresMap.getImagePoints(0, inlierPointPairs[imagePairID])
    image2Inliers = featuresMap.getImagePoints(imageID, inlierPointPairs[imagePairID])
    inlierGlobalIdxs = inlierPointPairs[imagePairID]

    intersectIdx = list(allImagePointIdx & set(inlierGlobalIdxs))
    intersectImagePoints = featuresMap.getImagePoints(imageID, intersectIdx)
    intersectWorldPoints = featuresMap.getWorldPoints(intersectIdx)
    Rinit, Cinit = PnPRANSAC(intersectWorldPoints, intersectImagePoints, K, 2000, 1)

    CNonlinear, RNonlinear = nonlinearPnPCR(K, intersectImagePoints, intersectWorldPoints, Cinit, Rinit)
    allC[imagePair] = CNonlinear
    allR[imagePair] = RNonlinear
    print(Rinit, Cinit)
    print(RNonlinear, CNonlinear)

    Xlinear = LinearTriangulation(K, image1Inliers, image2Inliers, np.eye(3), np.zeros((3,1)), RNonlinear, CNonlinear)
    XNonlinear = nonlinearTriangulation(K, np.eye(3), np.zeros((3,1)), RNonlinear, CNonlinear, image1Inliers, image2Inliers, Xlinear)
    featuresMap.updateWorldPoints(list(inlierGlobalIdxs), XNonlinear)
    allImagePointIdx = allImagePointIdx | set(inlierGlobalIdxs)
    drawWorldPoints([XNonlinear])

    matchedImageID = np.linspace(0, imageID, imageID + 1).astype(int)
    visMatrix = BuildVisibilityMatrix(matchedImageID, np.array(list(allImagePointIdx), dtype=int), featuresMap.visibilityMap)

    # before bundle adjustment
    imagePointMatrix = [featuresMap.getImagePoints(int(i), list(allImagePointIdx)) for i in matchedImageID]
    Xinit = featuresMap.getWorldPoints(list(allImagePointIdx))

    allCbundle, allRbundle, Xbundle = BundleAdjustment(allC[:imagePair+1], allR[:imagePair+1], Xinit, imagePointMatrix, K, visMatrix)
    allC[:imagePair+1] = allCbundle
    allR[:imagePair+1] = allRbundle
    featuresMap.updateWorldPoints(list(allImagePointIdx), Xbundle)
    drawWorldPoints([XNonlinear, Xbundle])
