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
from net_repo import EpiDataset, EpiveyorNet
from torchsummary import summary


checkpoint_path = 'media/Checkpoints'

net = EpiveyorNet(32, 1)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
net = net.to(device)

for param in net.parameters():
    param.requires_grad = True

lr = 0.001
optimizer = optim.Adam(net.parameters(), lr=lr)

criterion = nn.L1Loss()

dataset = EpiDataset(folder='/tmp/gino/')
dataset_test = EpiDataset(folder='/tmp/gino/')

training_generator = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
validation_generator = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

for epoch in range(5001):

    print("EPOCH", epoch)

    if epoch % 200 == 0 and epoch >0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(net.state_dict(), os.path.join(checkpoint_path,"saved_epoch_{}.pb".format(epoch)))
        torch.save(net.state_dict(), os.path.join(checkpoint_path, "last_model.pb"))

    if epoch % 200 == 0 and epoch > 0:
        lr = lr * 0.8
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("LEANING RAT CHANGED", "!" * 20)

    cumulative_loss = 0.0
    counter = 0.0
    for index,batch in enumerate(training_generator):
        net.train()
        optimizer.zero_grad()

        input = batch['rgb']
        target = batch['depth']
        mask = batch['mask']

        # print(input.shape)
        # target = target * mask
        #
        # target = target.cpu().numpy()
        # cv2.imshow("depth", (target[0][0]*255.).astype(np.uint8))
        # # cv2.imshow("mask", (mask[0][0] * 255.).astype(np.uint8))
        # cv2.waitKey(0)
        # import sys
        # sys.exit(0)

        input = input.to(device)
        target = target.to(device)
        mask = target.to(device)

        with torch.set_grad_enabled(True):
            # print("INPUT", input.shape)
            output = net(input)

            # print("OUTPUT", output.shape)

            # output_background = output * mask
            # target_background = target * mask
            #
            # output_foreground = output * (1.0 - mask)
            # target_foreground = target * (1.0 - mask)
            #
            # loss1 = criterion(output_background, target_background)
            # loss2 = criterion(output_foreground, target_foreground)
            # loss = loss1 + 1000 * loss2

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            cumulative_loss += loss.detach().cpu().numpy()
            counter += 1.0
            print("Batch: {}/{}".format(index, len(training_generator)))

    print("Loss", cumulative_loss / counter)

    #
    for batch in validation_generator:
        print("∞" * 20)
        print("TEST " * 20)
        net.eval()
        input = batch['rgb']
        target = batch['depth'].cpu().numpy()

        input = input.to(device)

        output = net(input).detach().cpu().numpy()

        map_gt = (target[0][0] * 255).astype(np.uint8)
        map_pred = (output[0][0] * 255).astype(np.uint8)

        # print("SHAPE", map_pred.shape)
        cv2.imwrite("/tmp/gt.png", map_gt)
        cv2.imwrite("/tmp/pred.png", map_pred)
        break
