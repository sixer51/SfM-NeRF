import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np

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
def drawMatchs(img1,img2,pairs):
    emptyImage = np.zeros((img2.shape[0],img1.shape[1],3), dtype=np.uint8)
    matchImage = np.concatenate((img2,emptyImage), axis=1)
    matchImage[0:img1.shape[0],img2.shape[1]:img2.shape[1]+img1.shape[1],...] = img1
    # img1FeaturePoints = img1FeaturePoints + np.array([0,len(img2[0])])
    for pair in pairs:
        leftPoint = (int(pair.coords2[0]), int(pair.coords2[1]))
        rightPoint = (int(pair.coords1[0]) + len(img2[0]), int(pair.coords1[1]))

        matchImage = cv2.circle(matchImage, (leftPoint[0],leftPoint[1]), radius = 4, color=(0,0,255), thickness = 2)
        matchImage = cv2.circle(matchImage, (rightPoint[0],rightPoint[1]), radius = 4, color=(0,255,0), thickness = 2)
        matchImage = cv2.line(matchImage, leftPoint, rightPoint, color=(255,0,0),thickness=2)
    return matchImage

def drawWorldPoints(multiXs, labels = []):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    for i in range(len(multiXs)):
        x = multiXs[i, :, 0:]
        y = multiXs[i, :, 1:]
        z = multiXs[i, :, 2:]
        ax.scatter(x, y, z)
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return