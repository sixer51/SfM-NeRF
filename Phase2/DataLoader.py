import numpy as np
import os
import torch
import json
import cv2

def loadDataset(basePath):
    datasetTypeList = ['train', 'val', 'test']
    datasetDict = {}
    count = 0
    split = []
    allImages = {}
    allPoses = {}
    for datasetType in datasetTypeList:
        fullPath = os.path.join(basePath, 'transforms_{}.json'.format(datasetType))
        with open(fullPath, 'r') as file:
            datasetDict[datasetType] = json.load(file)

        frames = datasetDict[datasetType]["frames"]
        images = []
        poses = []
        for frame in frames:
            imgName = os.path.join(basePath, frame['file_path'] + '.png')
            image = cv2.imread(imgName)
            images.append(image)
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)

        split.append(np.arange(count, count + len(images)))
        count += len(images)

        images = (np.array(images) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        allImages[datasetType] = images
        allPoses[datasetType] = poses

    height, width = images[0].shape[:2]
    camera_angle_x = float(datasetDict['train']['camera_angle_x'])
    focal = .5 * width / np.tan(.5 * camera_angle_x)
    K = np.array([
        [focal, 0, width / 2.],
        [0, focal, height / 2.],
        [0, 0, 1]
    ])

    near = 2.
    far = 6.

    return allImages, allPoses, [height, width, focal, K], near, far, split

# def getRenderPose(viewAngle=-30, interval=41, radius=4.0):

# images, poses, hwfk, near, far, split = loadDataset("./Phase2/data/lego/")
# print(hwfk)