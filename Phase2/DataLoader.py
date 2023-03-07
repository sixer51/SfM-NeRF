import numpy as np
import os
import torch
import json
import cv2

def loadDataset(datasetType, basePath):
    fullPath = os.path.join(basePath, 'transforms_{}.json'.format(datasetType))
    dataset = []
    with open(fullPath, 'r') as file:
        dataset = json.load(file)
    frames = dataset["frames"]

    images = []
    poses = []
    for frame in frames:
        imgName = os.path.join(basePath, frame['file_path'] + '.png')
        image = cv2.imread(imgName)
        images.append(image)
        pose = np.array(frame["transform_matrix"])
        poses.append(pose)

    images = (np.array(images) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = images[0].shape[:2]
    camera_angle_x = float(dataset['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([
        [focal, 0, W / 2.],
        [0, focal, H / 2.],
        [0, 0, 1]
    ])

    return H, W, focal, K, images, poses

# H, W, focal, K, images, poses = loadDataset("train", "./Phase2/data/lego/")
# print(H, W, focal, K)