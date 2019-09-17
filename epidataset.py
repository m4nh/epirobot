import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torch.optim as optim

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
from skimage import io, transform
import cv2
import types
import random

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class EpiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folder, max_depth=11, crop_size=128):

        self.folder = folder
        self.subfolders = sorted(glob.glob(os.path.join(self.folder, "*")))
        self.crop_size = crop_size
        self.max_depth = max_depth
        self.cache = {}
        self.image_caches = {}
        self.depth_caches = {}

        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ColorJitter(brightness=.4, contrast=0.3, saturation=1.0, hue=.5),
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        return len(self.subfolders)

    @staticmethod
    def displayableImage(image, depth=0):
        image2 = image.permute(2, 3, 0, 1)
        img = image2.cpu().numpy()[:, :, :, depth]
        return np.uint8(img * 255.)

    @staticmethod
    def displayableDepth(image, depth=0):
        img = image.cpu().numpy()[depth, :, :]
        img = 1. / img
        img = img / np.max(img)
        return np.uint8(img * 255.)

    @staticmethod
    def displayableEpiImage(image, y=None):
        image2 = image.permute(1, 3, 0, 2)
        if y is None:
            y = int(image.shape[3] / 2)
        img = image2.cpu().numpy()[:, :, :, y]
        return np.uint8(img * 255.)

    def loadImagesStack(self, folder):
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

        if folder not in self.image_caches:
            self.image_caches[folder] = []

            for index, image in enumerate(images):

                img = io.imread(image)
                img = torch.Tensor(np.float32(img) / 255.)
                img = img.permute(2, 0, 1)
                if index >= self.max_depth:
                    break
                self.image_caches[folder].append(img)

        return self.image_caches[folder]

    def loadDepthsStack(self, folder):
        images = sorted(glob.glob(os.path.join(folder, "*.exr")))

        if folder not in self.depth_caches:
            self.depth_caches[folder] = []

            for index, image in enumerate(images):

                img = cv2.imread(image, 2)
                img = torch.Tensor(1.0 / np.float32(img))
                # img = img.permute(2, 0, 1)
                if index >= self.max_depth:
                    break
                self.depth_caches[folder].append(img)

        return self.depth_caches[folder]

    def loadDepth(self, folder):

        images = sorted(glob.glob(os.path.join(folder, "*.exr")))

        depth = cv2.imread(images[0], 2)

        # depth = depth / np.max(depth)

        depth = np.expand_dims(depth, 0)
        return depth

    def buildMask(self, depth):
        mask = np.zeros(depth.shape, np.float32)
        mask[depth < 1.0] = 1.0

        return mask

    def getImageTransformPipeline(self, brightness = 0.5, contrast = 0.3, saturation = 0.2, hue = 0.3):
        # brightness = 0.3
        # contrast = 0.3
        # saturation = 1.0
        # hue = 0.5

        trans = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0.0, 1 - brightness), 1 + brightness)
            trans.append(Lambda(lambda img: FT.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0.0, 1 - contrast), 1 + contrast)
            trans.append(Lambda(lambda img: FT.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0.0, 1 - saturation), 1 + saturation)
            trans.append(Lambda(lambda img: FT.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            trans.append(Lambda(lambda img: FT.adjust_hue(img, hue_factor)))

        random.shuffle(trans)
        trans = [transforms.ToPILImage()] + trans + [transforms.ToTensor()]
        trans = transforms.Compose(trans)
        return trans

    def __getitem__(self, idx):

        trans = self.getImageTransformPipeline()

        # LOAD RBG IMAGES
        images = self.loadImagesStack(self.subfolders[idx])
        stack = []
        for im in images:
            imout = trans(im)
            stack.append(torch.unsqueeze(imout, 1))

        # stack = self.transform(stack)
        stack = torch.cat(stack, 1)

        # LOAD DEPTHS
        depths = self.loadDepthsStack(self.subfolders[idx])
        dstack = []
        for d in depths:
            dstack.append(torch.unsqueeze(d, 0))
        dstack = torch.cat(dstack)

        h = stack.shape[2]
        w = stack.shape[3]

        ri = np.random.randint(0, h - self.crop_size - 1)
        rj = np.random.randint(0, w - self.crop_size - 1)

        stack = stack[:, :, ri:ri + self.crop_size, rj:rj + self.crop_size]
        dstack = dstack[:, ri:ri + self.crop_size, rj:rj + self.crop_size]
        # if idx not in self.cache:
        #     self.cache[idx] = {}
        #     self.cache[idx]['rgb'] = transforms.ToPILImage(self.loadRGB(self.subfolders[idx]))
        #     # self.cache[idx]['depth'] = self.loadDepth(self.subfolders[idx])
        #     # self.cache[idx]['mask'] = self.buildMask(self.cache[idx]['depth'])

        # torchvision.transforms.ColorJitter
        sample = {
            'rgb': stack,
            'depth': dstack
        }

        return sample
