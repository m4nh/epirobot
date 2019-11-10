from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import os, glob
from resnet_models import BaseNetwork
from torch.utils.data import Dataset, DataLoader
import torchsummary
from skimage import io, transform
from skimage.transform import rescale
import torch.optim as optim
import cv2
from torch.autograd import Variable
from math import exp
from kornia.losses import SSIM
from torchvision import transforms, utils


class AnoDataset(Dataset):

    def __init__(self, folder, is_negative=False):
        self.folder = folder
        self.images = sorted(glob.glob(os.path.join(self.folder, '*')))
        self.is_negative = is_negative

        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.Grayscale(),
                transforms.RandomAffine(180, (0.02, 0.02),fillcolor=200),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.images)

    @staticmethod
    def displayableImage(image):
        img = image.permute(1, 2, 0)
        img = img.cpu().numpy()
        return np.uint8(img * 255.)

    def __getitem__(self, idx):
        # print(self.subfolders[idx])

        img = io.imread(self.images[idx])
        img = torch.Tensor(np.float32(img) / 255.)
        img = img.permute(2, 0, 1)

        input_image = self.tf(img)
        target_image = input_image.clone()

        if self.is_negative:
            target_image = torch.zeros_like(input_image)
        return {
            'input': input_image,
            'target': target_image
        }


dataset = AnoDataset(folder='/tmp/ano_dataset_train')
generator = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)

# LOAD MODEL IF
for d in dataset:
    img = dataset[0]['input']
    img = np.squeeze((img.numpy() * 255.).astype(np.uint8), 0)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    print(img.shape)
