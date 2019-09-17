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


class EpiveyorNet(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(EpiveyorNet, self).__init__()

        self.output_nc = output_nc

        self.layer_1_3D = self.convblock3D(input_nc, 1, 128, 5, 1)
        self.layer_1_2D = self.convblock2D(1, 128, 5)

        self.layer_2 = self.convblock2D(256, 256, 3, 2)
        self.layer_3 = self.convblock2D(256, 512, 3, 2)

        self.layer_up_3 = self.convblock2D(512, 256, 3, 1)
        self.layer_up_2 = self.convblock2D(512, 256, 3, 1)
        self.layer_up_1 = self.convblock2D(512, 128, 3, 1)

        self.layer_pred = self.lastblock(128, 1)

    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3, stride=2):
        block = []

        block += [nn.Conv3d(1, 128, kernel_size=(depth_dim, 5, 5), stride=1, padding=2)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm3d(out_dim)]
        block += [nn.Conv3d(128, 128, kernel_size=(5, 3, 3), stride=1, padding=0)]
        block += [nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))]

        return nn.Sequential(*block)

    def convblock2D(self, in_dim, out_dim, kernel=3, stride=1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=(kernel, kernel), stride=1,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def lastblock(self, in_dim, out_dim):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.Sigmoid()]

        return nn.Sequential(*block)

    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def scale_pyramid_(self, img, num_scales):
        # img = torch.mean(img, 1)
        # img = torch.unsqueeze(img, 1)
        scaled_imgs = [img]
        s = img.size()
        h = int(s[2])
        w = int(s[3])
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            temp = nn.functional.upsample(img, [nh, nw], mode='nearest')
            scaled_imgs.append(temp)
        return scaled_imgs

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_disparity_smoothness(self, disp, input_img):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in input_img]
        image_gradients_y = [self.gradient_y(img) for img in input_img]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

        smoothness_x = [torch.nn.functional.pad(k, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant') for k in smoothness_x]
        smoothness_y = [torch.nn.functional.pad(k, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant') for k in smoothness_y]

        return smoothness_x, smoothness_y

    def debugPrint(self, label, *argv):
        # debug = True
        # if debug: print(label.ljust(15), *argv)
        pass

    def forward(self, x):
        # 3x256x512
        debug = True

        self.debugPrint("Input:", x.shape)

        xu = torch.unsqueeze(x, 1)
        self.debugPrint("Input Unsq.:", xu.shape)

        l1_3d_u = self.layer_1_3D(xu)
        self.debugPrint("L1 3D Unsq.:", l1_3d_u.shape)

        l1_3d = torch.squeeze(l1_3d_u, 2)
        self.debugPrint("L1 3D:", l1_3d.shape)

        frame = torch.unsqueeze(x[:, 0, ::], 1)
        l1_2d = self.layer_1_2D(frame)
        self.debugPrint("L1 2D:", l1_2d.shape)

        l1 = torch.cat((l1_3d, l1_2d), 1)
        self.debugPrint("L1:", l1.shape)

        l2 = self.layer_2(l1)
        self.debugPrint("L2:", l2.shape)

        l3 = self.layer_3(l2)
        self.debugPrint("L3:", l3.shape)

        l3_up = self.layer_up_3(self.upsample_(l3, 2))
        self.debugPrint("L3 UP:", l3_up.shape)

        l3_up_plus = torch.cat((l3_up, l2), 1)
        self.debugPrint("L3 UP+:", l3_up_plus.shape)

        l2_up = self.layer_up_2(self.upsample_(l3_up_plus, 2))
        self.debugPrint("L2 UP:", l2_up.shape)

        l2_up_plus = torch.cat((l2_up, l1), 1)
        self.debugPrint("L2 UP+:", l2_up_plus.shape)

        l1_up = self.layer_up_1(l2_up_plus)
        self.debugPrint("L1 UP:", l1_up.shape)

        pred = self.layer_pred(l1_up)
        self.debugPrint("Pred:", pred.shape)

        return pred


class EpiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folder):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder = folder
        self.subfolders = sorted(glob.glob(os.path.join(self.folder, "*")))
        self.cache = {}

    def __len__(self):
        return len(self.subfolders)

    def loadRGB(self, folder):
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        images = list(map(cv2.imread, images))

        stack = None

        for index, img in enumerate(images):

            rand_3color = 0.05 + np.random.rand(3)
            rand_3color = rand_3color / np.sum(rand_3color)
            R = rand_3color[0]
            G = rand_3color[1]
            B = rand_3color[2]

            imgc = img.copy().astype(float)
            gray = B * imgc[:, :, 2] + G * imgc[:, :, 1] + R * imgc[:, :, 0]  # cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.uint8)

            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = np.expand_dims(gray, 0)
            # print(gray.shape)
            if stack is None:
                stack = gray
            else:
                # stack = np.dstack((stack, gray))
                stack = np.vstack((stack, gray))

        # stack = cv2.flip(stack, 0)
        stack = stack.astype(np.float32)
        stack = stack / 255.
        # print(stack.shape)
        return stack

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

    def __getitem__(self, idx):

        # if idx not in self.cache:
        self.cache[idx] = {}
        self.cache[idx]['rgb'] = self.loadRGB(self.subfolders[idx])
        self.cache[idx]['depth'] = self.loadDepth(self.subfolders[idx])
        self.cache[idx]['mask'] = self.buildMask(self.cache[idx]['depth'])

        sample = {
            'rgb': self.cache[idx]['rgb'],
            'depth': self.cache[idx]['depth'],
            'mask': self.cache[idx]['mask']
        }

        return sample


class EpiDatasetCrop(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folder, max_depth=16, crop_size=64):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder = folder
        self.subfolders = sorted(glob.glob(os.path.join(self.folder, "*")))
        self.crop_size = crop_size
        self.max_depth = max_depth
        self.cache = {}

    def __len__(self):
        return len(self.subfolders)

    def loadRGB(self, folder):
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        images = list(map(cv2.imread, images))

        stack = None

        for index, img in enumerate(images):

            if index >= self.max_depth:
                break
            # rand_3color = 0.05 + np.random.rand(3)
            # rand_3color = rand_3color / np.sum(rand_3color)
            # R = rand_3color[0]
            # G = rand_3color[1]
            # B = rand_3color[2]
            #
            # imgc = img.copy().astype(float)
            # gray = B * imgc[:, :, 2] + G * imgc[:, :, 1] + R * imgc[:, :, 0]  # cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # gray = gray.astype(np.uint8)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = np.expand_dims(gray, 0)
            # print(gray.shape)
            if stack is None:
                stack = gray
            else:
                # stack = np.dstack((stack, gray))
                stack = np.vstack((stack, gray))

        # stack = cv2.flip(stack, 0)
        stack = stack.astype(np.float32)
        stack = stack / 255.
        # print(stack.shape)
        return stack

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

    def __getitem__(self, idx):

        if idx not in self.cache:
            self.cache[idx] = {}
            self.cache[idx]['rgb'] = self.loadRGB(self.subfolders[idx])
            self.cache[idx]['depth'] = self.loadDepth(self.subfolders[idx])
            self.cache[idx]['mask'] = self.buildMask(self.cache[idx]['depth'])

        d, h, w = self.cache[idx]['rgb'].shape[:3]

        ri = np.random.randint(0, h - self.crop_size - 1)
        rj = np.random.randint(0, w - self.crop_size - 1)

        # self.cache[idx]['rgb'] = self.cache[idx]['rgb'][:, ri:ri + self.crop_size, rj:rj + self.crop_size]
        # self.cache[idx]['depth'] = self.cache[idx]['depth'][:, ri:ri + self.crop_size, rj:rj + self.crop_size]
        # self.cache[idx]['mask'] = self.cache[idx]['mask'][:, ri:ri + self.crop_size, rj:rj + self.crop_size]

        sample = {
            'rgb': self.cache[idx]['rgb'][:, ri:ri + self.crop_size, rj:rj + self.crop_size],
            'depth': self.cache[idx]['depth'][:, ri:ri + self.crop_size, rj:rj + self.crop_size],
            'mask': self.cache[idx]['mask'][:, ri:ri + self.crop_size, rj:rj + self.crop_size]
        }

        return sample


class EpiveyorPathNet(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(EpiveyorPathNet, self).__init__()

        self.output_nc = output_nc

        self.layer_1_3D = self.convblock3D(input_nc, 1, 128, 3, 1)
        self.layer_2_1 = self.convblock2D(128, 64, 7)
        self.layer_2_2 = self.convblock2D(128, 64, 5)
        self.layer_2_3 = self.convblock2D(128, 64, 3)
        self.layer_end = self.lastblock(192, 1)

    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3, stride=2):
        block = []

        block += [nn.Conv3d(1, out_dim, kernel_size=(depth_dim, 5, 5), stride=1, padding=2)]
        block += [nn.BatchNorm3d(out_dim)]
        block += [nn.LeakyReLU()]
        block += [nn.Conv3d(out_dim, out_dim, kernel_size=(5, 3, 3), stride=1, padding=0)]
        block += [nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))]
        # block += [nn.BatchNorm3d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def convblock2D(self, in_dim, out_dim, kernel=3, stride=1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.LeakyReLU()]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=(kernel, kernel), stride=1,
                            padding=int((kernel - 1) / 2))]
        # block += [nn.BatchNorm2d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def lastblock(self, in_dim, out_dim):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        # block += [nn.Sigmoid()]

        return nn.Sequential(*block)

    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        smoothness_x = torch.nn.functional.pad(smoothness_x, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        smoothness_y = torch.nn.functional.pad(smoothness_y, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def getTargetGradient(self, target):
        gx = torch.nn.functional.pad(self.gradient_x(target), (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        gy = torch.nn.functional.pad(self.gradient_y(target), (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')
        g = torch.abs(gx + gy)
        return g

    def debugPrint(self, label, *argv):
        # debug = True
        # if debug: print(label.ljust(15), *argv)
        pass

    def forward(self, x):
        # 3x256x512
        debug = True

        self.debugPrint("Input:", x.shape)

        xu = torch.unsqueeze(x, 1)
        self.debugPrint("Input Unsq.:", xu.shape)

        l1_3d_u = self.layer_1_3D(xu)
        self.debugPrint("L1 3D Unsq.:", l1_3d_u.shape)

        l1_3d = torch.squeeze(l1_3d_u, 2)
        self.debugPrint("L1 3D:", l1_3d.shape)

        l2_1 = self.layer_2_1(l1_3d)
        self.debugPrint("L2-1:", l2_1.shape)

        l2_2 = self.layer_2_2(l1_3d)
        self.debugPrint("L2-2:", l2_2.shape)

        l2_3 = self.layer_2_3(l1_3d)
        self.debugPrint("L2-3:", l2_3.shape)

        l2 = torch.cat((l2_1, l2_2, l2_3), 1)
        self.debugPrint("L2:", l2.shape)

        l_end = self.layer_end(l2)
        self.debugPrint("LEND:", l_end.shape)

        return l_end


class EpiveyorPatchNet2D(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(EpiveyorPatchNet2D, self).__init__()

        self.output_nc = output_nc

        self.layer_1_1 = self.convblock2D(input_nc, 256, 5, 2)
        self.layer_1_2 = self.convblock2D(input_nc, 256, 3, 2)
        self.layer_1_3 = self.convblock2D(input_nc, 256, 2, 2)
        self.layer_2 = self.convblock2D(256 * 3, 1024, 3, 2)
        self.layer_3 = self.convblock2D(1024, 512, 3)
        self.layer_4 = self.convblock2D(512, 256, 3)

        self.layer_end = self.lastblock(256, 1)
        # self.layer_2_1 = self.convblock2D(128, 64, 7)
        # self.layer_2_2 = self.convblock2D(128, 64, 5)
        # self.layer_2_3 = self.convblock2D(128, 64, 3)
        # self.layer_end = self.lastblock(192, 1)

    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3, stride=2):
        block = []

        block += [nn.Conv3d(1, out_dim, kernel_size=(depth_dim, 5, 5), stride=1, padding=2)]
        block += [nn.BatchNorm3d(out_dim)]
        block += [nn.LeakyReLU()]
        block += [nn.Conv3d(out_dim, out_dim, kernel_size=(5, 3, 3), stride=1, padding=0)]
        block += [nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))]
        # block += [nn.BatchNorm3d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def convblock2D(self, in_dim, out_dim, kernel=3, stride=1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.LeakyReLU()]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=(kernel, kernel), stride=1,
                            padding=int((kernel - 1) / 2))]
        # block += [nn.BatchNorm2d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def lastblock(self, in_dim, out_dim):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        # block += [nn.Sigmoid()]

        return nn.Sequential(*block)

    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        smoothness_x = torch.nn.functional.pad(smoothness_x, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        smoothness_y = torch.nn.functional.pad(smoothness_y, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def getTargetGradient(self, target):
        gx = torch.nn.functional.pad(self.gradient_x(target), (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        gy = torch.nn.functional.pad(self.gradient_y(target), (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')
        g = torch.abs(gx + gy)
        return g

    def debugPrint(self, label, *argv):
        # debug = True
        # if debug: print(label.ljust(15), *argv)
        pass

    def forward(self, x):
        # 3x256x512
        debug = True

        self.debugPrint("Input:", x.shape)

        l1_1 = self.layer_1_1(x)
        self.debugPrint("L1_1:", l1_1.shape)

        l1_2 = self.layer_1_2(x)
        self.debugPrint("L1_2:", l1_2.shape)

        l1_3 = torch.nn.functional.pad(self.layer_1_3(x), (1, 0, 1, 0), mode='replicate')
        # l1_3 = self.layer_1_3(x)
        self.debugPrint("L1_3:", l1_3.shape)

        l1 = torch.cat((l1_1, l1_2, l1_3), 1)
        self.debugPrint("L1:", l1.shape)

        l2 = self.layer_2(l1)
        self.debugPrint("L2:", l2.shape)

        l3 = self.layer_3(self.upsample_(l2, 2))
        self.debugPrint("L3:", l3.shape)

        l4 = self.layer_4(self.upsample_(l3, 2))
        self.debugPrint("L4:", l4.shape)

        l_end = self.layer_end(l4)
        self.debugPrint("LEND:", l_end.shape)
        # l1_3d = torch.squeeze(l1_3d_u, 2)
        # self.debugPrint("L1 3D:", l1_3d.shape)
        #
        # l2_1 = self.layer_2_1(l1_3d)
        # self.debugPrint("L2-1:", l2_1.shape)
        #
        # l2_2 = self.layer_2_2(l1_3d)
        # self.debugPrint("L2-2:", l2_2.shape)
        #
        # l2_3 = self.layer_2_3(l1_3d)
        # self.debugPrint("L2-3:", l2_3.shape)
        #
        # l2 = torch.cat((l2_1, l2_2, l2_3), 1)
        # self.debugPrint("L2:", l2.shape)
        #
        # l_end = self.layer_end(l2)
        # self.debugPrint("LEND:", l_end.shape)

        return l_end


class EpiRubikNet(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(EpiRubikNet, self).__init__()

        self.output_nc = output_nc

        self.layer_1_A = self.convblock2D(16, 256, 3)
        self.layer_1_B = self.convblock2D(16, 256, 3)
        self.layer_2 = self.convblock2D(512, 256, 3)
        self.layer_3 = self.convblock2D(256, 128, 3)
        self.layer_4 = self.convblock2D(128, 64, 3)
        self.layer_end = self.lastblock(64, 1)
        # self.layer_2_1 = self.convblock2D(128, 64, 7)
        # self.layer_2_2 = self.convblock2D(128, 64, 5)
        # self.layer_2_3 = self.convblock2D(128, 64, 3)
        # self.layer_end = self.lastblock(192, 1)

    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3, stride=2):
        block = []

        block += [nn.Conv3d(1, out_dim, kernel_size=(depth_dim, 5, 5), stride=1, padding=2)]
        block += [nn.BatchNorm3d(out_dim)]
        block += [nn.LeakyReLU()]
        block += [nn.Conv3d(out_dim, out_dim, kernel_size=(5, 3, 3), stride=1, padding=0)]
        block += [nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))]
        # block += [nn.BatchNorm3d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def convblock2D(self, in_dim, out_dim, kernel=3, stride=1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def lastblock(self, in_dim, out_dim):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        # block += [nn.Sigmoid()]

        return nn.Sequential(*block)

    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        smoothness_x = torch.nn.functional.pad(smoothness_x, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        smoothness_y = torch.nn.functional.pad(smoothness_y, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def getTargetGradient(self, target):
        gx = torch.nn.functional.pad(self.gradient_x(target), (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        gy = torch.nn.functional.pad(self.gradient_y(target), (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')
        g = torch.abs(gx + gy)
        return g

    def debugPrint(self, label, *argv):
        # debug = True
        # if debug: print(label.ljust(15), *argv)
        pass

    def forward(self, x):
        # 3x256x512
        debug = True

        x_A = x
        x_B = x.permute(0, 2, 1, 3)

        self.debugPrint("x_A:", x_A.shape)
        self.debugPrint("x_B:", x_B.shape)

        l1_A = self.layer_1_A(x_A)
        self.debugPrint("L1_A:", l1_A.shape)

        l1_B = self.layer_1_B(x_B)
        self.debugPrint("L1_B:", l1_B.shape)

        l1 = torch.cat((l1_A, l1_B), 1)
        self.debugPrint("L1:", l1.shape)

        l2 = self.layer_2(l1)
        self.debugPrint("L2:", l2.shape)
        l3 = self.layer_3(l2)
        self.debugPrint("L3:", l3.shape)
        l4 = self.layer_4(l3)
        self.debugPrint("L4:", l4.shape)

        l_end = self.layer_end(l4)
        self.debugPrint("LEND:", l_end.shape)

        return l_end


class UEpiNet(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(UEpiNet, self).__init__()

        self.output_nc = output_nc

        self.layer_1 = self.convblock3D(11, 3, 16, 3)
        self.layer_1_D = self.downconv3D(16, 32, 2)
        self.layer_2 = self.convblock3D(11, 32, 32, 3)
        self.layer_2_D = self.downconv3D(32, 64, 2)
        self.layer_3 = self.convblock3D(11, 64, 64, 3)
        self.layer_3_D = self.downconv3D(64, 64, 2)
        self.layer_4 = self.convblock3D(11, 64, 64, 3)

        self.layer_4_U = self.upconv3D(64, 64, 2)

        self.uplayer_3 = self.convblock3D(11, 64 + 64, 64, 3)
        self.uplayer_3_U = self.upconv3D(64, 32, 2)
        self.uplayer_2 = self.convblock3D(11, 64, 32, 3)
        self.uplayer_2_U = self.upconv3D(32, 16, 2)
        self.uplayer_1 = self.convblock3D(11, 32, 16, 3)

        self.last = self.endblock(11, 16, 1, 3)
        # self.uplayer_4 = self.convblock3D(11, 64, 64, 3, 2)

        # self.layer_5 = self.convblock3D(11, 64)

        # self.layer_3 = self.convblock2D(11, 32, 64)
        # self.layer_4 = self.convblock2D(128, 64, 3)
        # self.layer_end = self.lastblock(64, 1)
        # self.layer_2_1 = self.convblock2D(128, 64, 7)
        # self.layer_2_2 = self.convblock2D(128, 64, 5)
        # self.layer_2_3 = self.convblock2D(128, 64, 3)
        # self.layer_end = self.lastblock(192, 1)

    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3):
        block = []

        block += [nn.Conv3d(in_dim, out_dim, kernel_size=(kernel, kernel, kernel), stride=1, padding=1)]
        block += [nn.PReLU()]
        block += [nn.BatchNorm3d(out_dim)]

        block += [nn.Conv3d(out_dim, out_dim, kernel_size=(kernel, kernel, kernel), stride=1, padding=1)]
        block += [nn.PReLU()]

        # block += [nn.Conv3d(out_dim, out_dim, kernel_size=(kernel, kernel, kernel), stride=1, padding=1)]
        # block += [nn.PReLU()]
        #
        # block += [nn.Conv3d(out_dim, out_dim, kernel_size=(kernel, kernel, kernel), stride=1, padding=1)]
        # block += [nn.PReLU()]

        return nn.Sequential(*block)

    def downconv3D(self, in_dim, out_dim, stride=2):
        block = [
            nn.Conv3d(in_dim, out_dim, kernel_size=(1, stride, stride), stride=(1, stride, stride),
                      padding=(0, 0, 0))]

        return nn.Sequential(*block)

    def upconv3D(self, in_dim, out_dim, stride=2):
        block = [
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, stride, stride), stride=(1, stride, stride),
                               padding=(0, 0, 0))]

        return nn.Sequential(*block)

    def endblock(self, depth_dim, in_dim, out_dim, kernel=3):
        block = []

        block += [nn.Conv3d(in_dim, out_dim, kernel_size=(kernel, kernel, kernel), stride=1, padding=1)]
        # block += [nn.BatchNorm3d(out_dim)]
        # block += [nn.LeakyReLU()]

        return nn.Sequential(*block)


    def upsample_(self, disp, ratio):
        s = disp.size()
        h = int(s[2])
        w = int(s[3])
        nh = h * ratio
        nw = w * ratio
        temp = nn.functional.upsample(disp, [nh, nw], mode='nearest')

        return temp

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        smoothness_x = torch.nn.functional.pad(smoothness_x, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        smoothness_y = torch.nn.functional.pad(smoothness_y, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def getTargetGradient(self, target):
        gx = torch.nn.functional.pad(self.gradient_x(target), (0, 1, 0, 0, 0, 0, 0, 0), mode='constant')
        gy = torch.nn.functional.pad(self.gradient_y(target), (0, 0, 0, 1, 0, 0, 0, 0), mode='constant')
        g = torch.abs(gx + gy)
        return g

    def debugPrint(self, label, *argv):
        debug = True
        if debug: print(label.ljust(15), *argv)
        pass

    def forward(self, x):
        # 3x256x512
        debug = True

        self.debugPrint("x:", x.shape)

        l1 = self.layer_1(x)
        self.debugPrint("L1:", l1.shape)

        l1_D = self.layer_1_D(l1)
        self.debugPrint("L1 DOWN:", l1_D.shape)

        l2 = self.layer_2(l1_D)
        self.debugPrint("L2:", l2.shape)

        l2_D = self.layer_2_D(l2)
        self.debugPrint("L2 DOWN:", l2_D.shape)

        l3 = self.layer_3(l2_D)
        self.debugPrint("L3:", l3.shape)

        l3_D = self.layer_3_D(l3)
        self.debugPrint("L3 DOWN:", l3_D.shape)

        l4 = self.layer_4(l3_D)
        self.debugPrint("L4:", l4.shape)

        l4_U = self.layer_4_U(l4)
        self.debugPrint("L4 UP:", l4_U.shape)

        ul3 = self.uplayer_3(torch.cat((l3, l4_U), 1))
        self.debugPrint("UL3:", ul3.shape)

        ul3_u = self.uplayer_3_U(ul3)
        self.debugPrint("UL3 UP:", ul3_u.shape)

        ul2 = self.uplayer_2(torch.cat((l2, ul3_u), 1))
        self.debugPrint("UL2:", ul2.shape)

        ul2_u = self.uplayer_2_U(ul2)
        self.debugPrint("UL2 UP:", ul2_u.shape)
        self.debugPrint("XXX:", torch.cat((l1, ul2_u), 1).shape)

        ul1 = self.uplayer_1(torch.cat((l1, ul2_u), 1))
        self.debugPrint("UL1 :", ul1.shape)

        last = self.last(ul1)
        self.debugPrint("LAST :", last.shape)

        return torch.squeeze(last, 1)
