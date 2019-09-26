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
import torchsummary
from resnet_models import ResnetModel, EpiFeatures, EpiNet, EpinetSimple
from epidataset import EpiDisparityDataset, EpiSingleDisparityDataset

import sys

learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
print("LEARNING RATE ", learning_rate)
checkpoint_path = 'media/Checkpoints'

# MODEL
model = EpinetSimple(11, 3, 1, "epi_simple_0", "media/Checkpoints/")

for param in model.parameters():
    param.requires_grad = True

# OPTIMIZER
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

# DATASET
dataset = EpiSingleDisparityDataset(folder='/Users/daniele/Desktop/EpiRobot/Videos/real_dataset', crop_size=460, focal=2000, max_disparity=255)

generator = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)



device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
model = model.to(device)

# LOAD MODEL IF ANY
model.loadModel(device=device)

torchsummary.summary(model, (3, 11, 32, 32), device="cuda" if torch.cuda.is_available() else "cpu")

for g in generator:
    input = g['rgb']
    input = input.to(device)

    output = model(input)


    rgb = dataset.displayableImage(input[0],5)
    depth = dataset.displayableDepth(output[0],0)
    print(rgb.shape, depth.shape)

    cv2.imshow("rgb", rgb)
    cv2.imshow("depth", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    cv2.waitKey(0)
