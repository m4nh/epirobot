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
from epidataset import EpiDisparityDataset

import sys

learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
print("LEARNING RATE ", learning_rate)
checkpoint_path = 'media/Checkpoints'

# MODEL
model = EpinetSimple(11, 3,1, "epi_simple_0", "media/Checkpoints/")


torchsummary.summary(model, (3, 11, 32, 32))
