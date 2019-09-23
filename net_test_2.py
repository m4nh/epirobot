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
from resnet_models import ResnetModel, EpiFeatures, EpiNet
from epidataset import EpiDisparityDataset
import sys

# DATASET
dataset = EpiDisparityDataset(folder='/tmp/train', crop_size=-1, focal=2000, max_disparity=255)
dataset_test = EpiDisparityDataset(folder='/tmp/test', crop_size=-1, augmentation=False)

training_generator = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
validation_generator = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

criterion = nn.L1Loss()

for index, batch in enumerate(training_generator):
    bd = batch['depth']
    depth = dataset.displayableDepth(bd[0], 5)

    print(np.min(depth), np.max(depth))
    cv2.imshow("image", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    cv2.waitKey(0)
