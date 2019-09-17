#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, glob
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from net_repo import EpiDatasetCrop, EpiveyorNet, EpiRubikNet,EpiveyorPathNet
from torchsummary import summary
import time
from collections import namedtuple


def tileImage(image, tile_size=(512, 512)):
    tileSizeX = tile_size[0]
    tileSizeY = tile_size[1]
    numTilesX = int(np.ceil(image.shape[2] / tileSizeX))
    numTilesY = int(np.ceil(image.shape[1] / tileSizeY))
    makeLastPartFull = True

    tiles = []
    Tile = namedtuple('Tile', 'x y w h data')

    for nTileX in range(numTilesX):
        for nTileY in range(numTilesY):
            startX = nTileX * tileSizeX
            endX = startX + tileSizeX
            startY = nTileY * tileSizeY
            endY = startY + tileSizeY

            if (endY > image.shape[1]):
                endY = image.shape[1]

            if (endX > image.shape[2]):
                endX = image.shape[2]

            if (makeLastPartFull == True and (nTileX == numTilesX - 1 or nTileY == numTilesY - 1)):
                startX = endX - tileSizeX
                startY = endY - tileSizeY

            currentTile = image[:, startY:endY, startX:endX]
            tiles.append(Tile(startX, startY, tileSizeX, tileSizeY, currentTile.copy()))
    return tiles


checkpoint_path = 'media/Checkpoints'

net = EpiveyorPathNet(16, 1)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
net = net.to(device)
map_location = 'cpu'

last_model_path = os.path.join(checkpoint_path, "last_model.pb")
if os.path.exists(last_model_path):
    net.load_state_dict(torch.load(last_model_path, map_location=device))
    print("*" * 10, "MODEL LOADED!", "*" * 10)

sample_path = '/private/tmp/powerlog/'
#sample_path = '/private/tmp/gino_test/frame_46192a90ffb44cc6ab227e3c826d91a9'
images = sorted(glob.glob(os.path.join(sample_path, '*')))
images = map(cv2.imread, images)
images = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in images]

dataset = EpiDatasetCrop(folder=sample_path)

volume = torch.Tensor(dataset.loadRGB(sample_path))
volume = nn.functional.upsample(torch.unsqueeze(volume,0), [240, 320], mode='bilinear')
print(volume.shape)

chunks = tileImage(np.squeeze(volume.numpy()), (32, 32))

net.eval()
depth = np.zeros((volume.shape[2], volume.shape[3]), np.float32)
for c in chunks:
    input = torch.Tensor(np.expand_dims(c.data, 0))


    print(input.shape)

    out = net(input).detach().cpu().numpy()
    print(c.y,c.x,c.h,c.w, c.data.shape)
    depth[c.y:c.y + c.h, c.x:c.x + c.w] = out


cv2.imshow("image", np.uint8(volume[0,0,::]*255))
cv2.imshow("depth", np.uint8(depth*255.))
cv2.waitKey(0)
