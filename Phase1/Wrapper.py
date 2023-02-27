from DataLoader import *
from Visualization import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from GetInliersRANSAC import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
import os

# Don't generate pyc codes
sys.dont_write_bytecode = True

dataDirPath = os.getcwd() + '/P3Data/'
images = loadImages(dataDirPath)

K = getCameraParams(dataDirPath)
print(K)

featuresMap = FeaturesAssociationMap(dataDirPath)
image1Coords, image2Coords, featureIdxs = featuresMap.get_feature_matches((1,2))

matchId = 0
image1Id = 0
image2Id = 1
InlierIdxs, OutlierIdxs = homographyRANSAC(image1Coords, image2Coords, 2000, 50)
image1Inliers_homography, image2Inliers_homography, featureIdxs_homography = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, InlierIdxs)
image1Outliers_homography, image2Outliers_homography, _ = filterCoordsByIdxs(image1Coords, image2Coords, featureIdxs, OutlierIdxs)
InlierIdxs, OutlierIdxs = ransac(image1Inliers_homography, image2Inliers_homography, 10000, 0.01)
image1Inliers, image2Inliers, featureIdxs = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, InlierIdxs)
image1Outliers, image2Outliers, _ = filterCoordsByIdxs(image1Inliers_homography, image2Inliers_homography,featureIdxs_homography, OutlierIdxs)

matchImage0 = drawMatchs(images[image1Id], images[image2Id], image1Outliers_homography, image2Outliers_homography, (255,0,0))
matchImage1 = drawMatchs(images[image1Id], images[image2Id], image1Inliers_homography, image2Inliers_homography, (0,255,0))
matchImage2 = drawMatchs(images[image1Id], images[image2Id], image1Outliers, image2Outliers, (255,0,0))
matchImage3 = drawMatchs(images[image1Id], images[image2Id], image1Inliers, image2Inliers, (0,255,0))
plt.figure(figsize=(16, 8))
plt.axis("off")
plt.imshow(matchImage0)
plt.show()
plt.figure(figsize=(16, 8))
plt.axis("off")
plt.imshow(matchImage1)
plt.show()
plt.figure(figsize=(16, 8))
plt.axis("off")
plt.imshow(matchImage2)
plt.show()
plt.figure(figsize=(16, 8))
plt.axis("off")
plt.imshow(matchImage3)
plt.show()

x1s = []
x2s = []
for idx in range(len(image1Inliers)):
    x1s.append(image1Inliers[idx])
    x2s.append(image2Inliers[idx])
x1s = np.array(x1s)
x2s = np.array(x2s)

Fest = EstimateFundamentalMatrix(x1s, x2s)
print(Fest)

Eest = EssentialFromFundamental(Fest, K)
print(Eest)

Rest, Test = extractCameraPose(Eest)
print(Rest)
print(Test)

Xest = []
for i in range(len(Rest)):
    X = LinearTriangulation(K, x1s, x2s, np.eye(3), np.zeros((3,1)), Rest[i], Test[i])
    Xest.append(X)

drawWorldPoints(Xest)
Rest, Test, Xest = disambiguateCameraPose(Rest, Test, Xest)
drawWorldPoints([Xest])

newImage = drawPoints(images[image1Id], x1s, [255, 0, 0])
reprojPoints = reprojection(K, np.eye(3), np.zeros((3,1)), Xest)
newImage = drawPoints(newImage, reprojPoints, [0, 255, 0])
plt.imshow(newImage)
plt.show()

newImage2 = drawPoints(images[image2Id], x2s, [255, 0, 0])
reprojPoints2 = reprojection(K, Rest, Test, Xest)
newImage2 = drawPoints(newImage2, reprojPoints2, [0, 255, 0])
plt.imshow(newImage2)
plt.show()

XestNonlinear = nonlinearTriangulation(K, np.eye(3), np.zeros((3,1)), Rest, Test, x1s, x2s, Xest)

newImage = drawPoints(images[image1Id], x1s, [255, 0, 0])
reprojPoints = reprojection(K, np.eye(3), np.zeros((3,1)), XestNonlinear)
newImage = drawPoints(newImage, reprojPoints, [0, 255, 0])
plt.imshow(newImage)
plt.show()