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
import os

# Don't generate pyc codes
sys.dont_write_bytecode = True

dataDirPath = os.getcwd() + '/P3Data/'
images = loadImages(dataDirPath)

K = getCameraParams(dataDirPath)
print(K)

featuresMap = FeaturesAssociationMap(dataDirPath)
image1Coords, image2Coords, featureIdxs = featuresMap.get_feature_matches((1,2))

imagePairs = [[0, 1], [0, 2], [0, 3], [0, 4]]
# pairInliners
AllImagePointIdx = set()

for i in range(len(imagePairs)):
    image1Id = imagePairs[i, 0]
    image2Id = imagePairs[i, 1]
    image1Coords, image2Coords, featureIdxs = featuresMap.get_feature_matches((image1Id,image2Id))
    InlierIdxs, OutlierIdxs = homographyRANSAC(image1Coords, image2Coords, 2000, 50)
    image1Inliers_homography, image2Inliers_homography, featureIdxs_homography = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, InlierIdxs)
    image1Outliers_homography, image2Outliers_homography, _ = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, OutlierIdxs)
    InlierIdxs, OutlierIdxs = ransac(image1Inliers_homography, image2Inliers_homography, 10000, 0.01)
    
    image1Inliers, image2Inliers, featureIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, InlierIdxs)
    image1Outliers, image2Outliers, OutlierGlobalIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, OutlierIdxs)
    featuresMap.removeMatches(OutlierGlobalIdxs, image2Id)
    # AllImagePointIdx = set(featureIdxs)

x1s = image1Inliers
x2s = image2Inliers

Fest = EstimateFundamentalMatrix(x1s, x2s)
Eest = EssentialFromFundamental(Fest, K)
Rest, Test = extractCameraPose(Eest)

Xest = []
for i in range(len(Rest)):
    X = LinearTriangulation(K, x1s, x2s, np.eye(3), np.zeros((3,1)), Rest[i], Test[i])
    Xest.append(X)

drawWorldPoints(Xest)
Rest, Test, Xest = disambiguateCameraPose(Rest, Test, Xest)
drawWorldPoints([Xest])

XestNonlinear = nonlinearTriangulation(K, np.eye(3), np.zeros((3,1)), Rest, Test, x1s, x2s, Xest)
featuresMap.updateWorldPoints(list(AllImagePointIdx), XestNonlinear)
drawWorldPoints([XestNonlinear])

image1Id = 0
image2Id = 2
image1Coords, image2Coords, featureIdxs = featuresMap.get_feature_matches((image1Id,image2Id))
InlierIdxs, OutlierIdxs = homographyRANSAC(image1Coords, image2Coords, 2000, 50)
image1Inliers_homography, image2Inliers_homography, featureIdxs_homography = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, InlierIdxs)
image1Outliers_homography, image2Outliers_homography, _ = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, OutlierIdxs)
InlierIdxs, OutlierIdxs = ransac(image1Inliers_homography, image2Inliers_homography, 10000, 0.01)
image1Inliers, image2Inliers, featureIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, InlierIdxs)
image1Outliers, image2Outliers, OutlierGlobalIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, OutlierIdxs)
featuresMap.removeMatches(OutlierGlobalIdxs, image2Id)
intersectIdx = AllImagePointIdx & set(featureIdxs)

intersectIdx = list(AllImagePointIdx & set(featureIdxs))
intersectImagePoints = np.vstack([featuresMap.featuresU[intersectIdx, image2Id], featuresMap.featuresV[intersectIdx, image2Id]]).T
intersectWorldPoints = featuresMap.worldPoints[intersectIdx]
Rinit, Cinit = PnPRANSAC(intersectWorldPoints, intersectImagePoints, K, 2000, 1)

CNonlinear, RNonlinear = nonlinearPnPCR(K, intersectImagePoints, intersectWorldPoints, Cinit, Rinit)
print(Rinit, Cinit)
print(RNonlinear, CNonlinear)

X13 = LinearTriangulation(K, image1Inliers, image2Inliers, np.eye(3), np.zeros((3,1)), RNonlinear, CNonlinear)
X13Nonlinear = nonlinearTriangulation(K, np.eye(3), np.zeros((3,1)), RNonlinear, CNonlinear, image1Inliers, image2Inliers, X13)

featuresMap.updateWorldPoints(list(featureIdxs), X13Nonlinear)
AllImagePointIdx = AllImagePointIdx | set(featureIdxs)

visMatrix = BuildVisibilityMatrix(np.array([0, 1, 2]), np.array(list(AllImagePointIdx), dtype=int), featuresMap.visibilityMap)