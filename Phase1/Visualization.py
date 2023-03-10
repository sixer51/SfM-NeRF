import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import copy

# Don't generate pyc codes
sys.dont_write_bytecode = True

def displayImages(images, numCol = 4):
    aspectRatio = images[0].shape[0]/images[0].shape[1]
    # plt.figure(figsize=(16/aspectRatio,16))
    numFigures = len(images)
    rowNum = numFigures//numCol + numCol
    fig = plt.figure(figsize=(numCol*64/rowNum/aspectRatio,numCol*64/numCol))
    

    ax = [fig.add_subplot(rowNum,numCol,i+1) for i in range(numFigures)]

    for i,a in enumerate(ax):
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
        a.imshow(images[i],aspect=1)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    return

# def drawMatchs(img1,img2,img1FeaturePoints, img2FeaturePoints, matches):
def drawMatchs(img1,img2, image1Coords, image2Coords,lineColor):
    emptyImage = np.zeros((img2.shape[0],img1.shape[1],3), dtype=np.uint8)
    matchImage = np.concatenate((img2,emptyImage), axis=1)
    matchImage[0:img1.shape[0],img2.shape[1]:img2.shape[1]+img1.shape[1],...] = img1
    # img1FeaturePoints = img1FeaturePoints + np.array([0,len(img2[0])])
    for i in range(len(image1Coords)):
        leftPoint = (int(image1Coords[i][0]), int(image1Coords[i][1]))
        rightPoint = (int(image2Coords[i][0]) + len(img2[0]), int(image2Coords[i][1]))

        matchImage = cv2.circle(matchImage, (leftPoint[0],leftPoint[1]), radius = 4, color=(0,0,255), thickness = 2)
        matchImage = cv2.circle(matchImage, (rightPoint[0],rightPoint[1]), radius = 4, color=(0,255,0), thickness = 2)
        matchImage = cv2.line(matchImage, leftPoint, rightPoint, color=lineColor,thickness=2)
    return matchImage

def drawPoints(img, points, color):
    newImage = copy.copy(img)
    for point in points:
        newImage = cv2.circle(newImage, (int(point[0]), int(point[1])), radius = 1, color=color, thickness = 2)
    return newImage


def drawWorldPoints(multiXs, labels = []):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax = plt.axes(projection ='3d')

    for i in range(len(multiXs)):
        x = multiXs[i][:, 0]
        # y = multiXs[i][:, 1]
        z = multiXs[i][:, 2]
        # ax.scatter(x, y, z, marker=".")
        ax.plot(x,z,".",markersize=1)
        
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    # ax.set_zlim(-200,200)
    plt.show()
    return

def drawWorldPoints3D(multiXs, labels = []):
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    ax = plt.axes(projection ='3d')

    for i in range(len(multiXs)):
        x = multiXs[i][:, 0]
        y = multiXs[i][:, 1]
        z = multiXs[i][:, 2]
        ax.scatter(x, y, z, marker=".")
        # ax.plot(x,z,".",markersize=1)
        
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.set_xlim(-200,200)
    ax.set_ylim(-200,200)
    ax.set_zlim(-200,200)
    plt.show()
    return