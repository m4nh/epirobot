import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import glob
import os

from epidataset import EpiDataset

dataset = EpiDataset(folder='/tmp/gino')

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

index = 0
while True:

    d = dataset[index]
    data = d['rgb']
    depth_data = d['depth']
    print("RGB", data.shape)
    print("DEPTH", depth_data.shape)

    for i in range(11):
        print(i)
        img = EpiDataset.displayableImage(data, depth=i)
        depth = EpiDataset.displayableDepth(depth_data, depth=i)
        print(img.shape)
        print(depth.shape)
        cv2.imwrite("/tmp/art.png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("depth", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.waitKey(0)

    index = (index +1)% len(dataset)



while True:
    for d in dataloader:
        print(d['rgb'].shape)
        image = d['rgb'][0]
        print(image.shape)

        # for y in range(50):
        #     img = EpiDataset.displayableEpiImage(image, y=y)
        #     print(img.shape)
        #     cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)

        img = EpiDataset.displayableImage(image)
        print(img.shape)
        cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
