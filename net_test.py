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

learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
print("LEARNING RATE ", learning_rate)
checkpoint_path = 'media/Checkpoints'

# MODEL
model = EpiNet(11, 1, "epi0", "media/Checkpoints/")
torchsummary.summary(model, (3, 11, 256, 256))

for param in model.parameters():
    param.requires_grad = True

# OPTIMIZER
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

# DATASET
dataset = EpiDisparityDataset(folder='/tmp/test', crop_size=256)
dataset_test = EpiDisparityDataset(folder='/tmp/test', crop_size=256)

training_generator = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
validation_generator = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

# LOAD MODEL IF ANY
model.loadModel()

criterion = nn.L1Loss()

for epoch in range(50001):

    print("EPOCH", epoch)

    if epoch % 10 == 0 and epoch > 0:
        model.saveModel()

    # CHANGE LEARNING RATE
    if epoch % 2000 == 0 and epoch > 0:
        lr = lr * 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("LEANING RAT CHANGED", "!" * 20)

    loss_ = 0.0
    counter = 0.0
    for index, batch in enumerate(training_generator):
        model.train()
        optimizer.zero_grad()

        input = batch['rgb']
        target = batch['depth']

        input = input.to(model.device)
        target = target.to(model.device)

        with torch.set_grad_enabled(True):
            output, output_2, output_4, output_8 = model(input)

            target = torch.unsqueeze(target[:, 5, :, :], 1)

            loss = model.buildLoss(output, target)

            loss.backward()
            optimizer.step()

            loss_ += loss.detach().cpu().numpy()
            counter += 1.0
            print("Batch: {}/{}".format(index, len(training_generator)))

    print("Loss", loss_ / counter)

    if True:  # epoch % 5 == 0 and epoch > 0:
        stack = None
        max_stack = 8
        print("∞" * 20)
        print("TEST " * 20)
        for index, batch in enumerate(validation_generator):

            model.eval()
            input = batch['rgb']
            target = batch['depth'].detach()

            input = input.to(model.device)

            output, output_2, output_4, output_8 = model(input)
            output = output.detach()

            # print("TG", target[0].shape, np.min(target[0].cpu().numpy()), np.max(target[0].cpu().numpy()))
            # print("OPUT", output[0].shape, np.min(output[0].cpu().numpy()), np.max(output[0].cpu().numpy()))

            map_gt = cv2.applyColorMap(dataset.displayableDepth(target[0], 5), cv2.COLORMAP_JET)
            map_pred = cv2.applyColorMap(dataset.displayableDepth(output[0]), cv2.COLORMAP_JET)

            map = np.vstack((input[:, 5, :, :], map_gt, map_pred))

            if stack is None:
                stack = map
            else:
                stack = np.hstack((stack, map))

            index += 1
            if index >= max_stack:
                break

        # print("SHAPE", map_pred.shape)
        cv2.imwrite("/tmp/predictions.png", stack)
