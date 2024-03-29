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


class mono_net(nn.Module):  # vgg version
    def __init__(self, input_nc, output_nc):
        super(mono_net, self).__init__()

        self.output_nc = output_nc

        self.layer_1 = self.convblock3D(input_nc, 1, 128, 2)
        self.layer_2 = self.convblock2D(128, 256, 3,2)

        self.layer_3 = self.convblock2D(256, 128, 3, 1)
        self.layer_4 = self.convblock2D(256, 64, 3, 1)

        self.layer_5 = self.lastblock(64, 1)


    def convblock3D(self, depth_dim, in_dim, out_dim, kernel=3,stride=2):
        block = []

        block += [nn.Conv3d(in_dim, out_dim, kernel_size=(depth_dim, kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm3d(out_dim)]

        return nn.Sequential(*block)

    def convblock2D(self, in_dim, out_dim, kernel=3, stride = 1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=(kernel, kernel), stride=1,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]

        return nn.Sequential(*block)

    def convblock2D_inv(self, in_dim, out_dim, kernel=3, stride = 1):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=(kernel, kernel), stride=stride,
                            padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]

        return nn.Sequential(*block)

    def lastblock(self, in_dim, out_dim):
        block = []

        block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.LeakyReLU()]
        block += [nn.BatchNorm2d(out_dim)]
        block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        block += [nn.Sigmoid()]

        return nn.Sequential(*block)

    def conv_down_block(self, in_dim, out_dim, kernal):
        conv_down_block = []
        conv_down_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernal, stride=1, padding=int((kernal - 1) / 2)),
                            nn.BatchNorm2d(out_dim), nn.ELU()]  # h,w -> h,w
        conv_down_block += [nn.Conv2d(out_dim, out_dim, kernel_size=kernal, stride=2, padding=int((kernal - 1) / 2)),
                            nn.BatchNorm2d(out_dim), nn.ELU()]  # h,w -> h/2,w/2

        return nn.Sequential(*conv_down_block)

    def conv_down_block3D(self, in_dim, out_dim, kernal):
        conv_down_block = []
        conv_down_block += [
            nn.Conv3d(1, out_dim, kernel_size=(in_dim, kernal, kernal), stride=2, padding=int((kernal - 1) / 2)),
            nn.BatchNorm3d(out_dim), nn.ELU()]  # h,w -> h,w

        return nn.Sequential(*conv_down_block)

    def conv_up_block(self, in_dim, out_dim):
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
                          nn.ELU()]  # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def conv_block(self, in_dim, out_dim):
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
                          nn.ELU()]  # h,w -> h,w

        return nn.Sequential(*conv_up_block)

    def disp_block(self, in_dim):
        disp_block = []
        disp_block += [nn.Conv2d(in_dim, self.output_nc, kernel_size=3, stride=1, padding=1),
                       nn.Sigmoid()]  # h,w -> h,w

        return nn.Sequential(*disp_block)

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

    def forward(self, x):
        # 3x256x512

        x = torch.unsqueeze(x, 1)
        # print("X:", x.shape)


        x = self.layer_1(x)
        # print("L:", x.shape)

        x = torch.squeeze(x)
        l = x
        # print("L:", x.shape)

        x = self.layer_2(x)
        # print("L:", x.shape)

        x = self.upsample_(x,2)
        # print("L:", x.shape)

        x = self.layer_3(x)
        # print("L:", x.shape)

        x = torch.cat((x,l),1)
        # print("CL:", x.shape)

        x = self.upsample_(x,2)
        # print("L:", x.shape)

        x = self.layer_4(x)
        # print("L:", x.shape)


        x = self.layer_5(x)
        # print("L:", x.shape)

        return x


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


checkpoint_path = 'media/Checkpoints'

net = mono_net(32, 1)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
net = net.to(device)

net.load_state_dict(torch.load(os.path.join(checkpoint_path, "last_model.pb"),map_location=device))
net.eval()


dataset = EpiDataset(folder='/tmp/gino/')
training_generator = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=False)

for index,batch in enumerate(training_generator):
    input = batch["rgb"].to(device)
    target = batch["depth"].to(device)

    print(input.shape)

    output = net(input)

    print(output.shape)


    out = output[0][0].detach().cpu().numpy()
    gt = target[0][0].cpu().numpy()

    out = (out*255).astype(np.uint8)
    gt = (gt * 255).astype(np.uint8)
    cv2.imshow("image", out)
    cv2.imshow("gt",gt)
    cv2.waitKey(0)


    break

