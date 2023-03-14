import numpy as np
import os
import torch
import json
import cv2
import imageio

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def getRenderPose():
    return torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

def loadDataset(basePath, halfRes=True, testskip=1):
    datasetTypeList = ['train', 'val', 'test']
    datasetDict = {}
    count = 0
    split = []
    # allImages = {}
    # allPoses = {}
    allImages = []
    allPoses = []
    for datasetType in datasetTypeList:
        fullPath = os.path.join(basePath, 'transforms_{}.json'.format(datasetType))
        with open(fullPath, 'r') as file:
            datasetDict[datasetType] = json.load(file)

        skip = 1 if datasetType == "train" else testskip
        frames = datasetDict[datasetType]["frames"][::skip]
        images = []
        poses = []
        for frame in frames:
            imgName = os.path.join(basePath, frame['file_path'] + '.png')
            image = imageio.imread(imgName)
            # image = cv2.imread(imgName)
            images.append(image)
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)

        split.append(np.arange(count, count + len(images)))
        count += len(images)

        images = (np.array(images) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        allImages.append(images)
        allPoses.append(poses)

    allImages = np.concatenate(allImages, 0)
    allPoses = np.concatenate(allPoses, 0)

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

    if halfRes:
        height = height//2
        width = width//2
        focal = focal/2.
        K = K/2.

        imgHalfRes = np.zeros((allImages.shape[0], height, width, 4))
        for i, img in enumerate(allImages):
            imgHalfRes[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            # images = (np.array(images) / 255.).astype(np.float32)
            # allImages[datasetType] = images
        allImages = imgHalfRes

    return allImages, allPoses, [height, width, focal], K, near, far, split