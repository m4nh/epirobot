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

        self.downconv_1 = self.conv_down_block3D(input_nc, 32, 7)
        self.downconv_2 = self.conv_down_block(32, 64, 5)
        self.downconv_3 = self.conv_down_block(64, 128, 3)
        self.downconv_4 = self.conv_down_block(128, 256, 3)
        self.downconv_5 = self.conv_down_block(256, 512, 3)
        self.downconv_6 = self.conv_down_block(512, 512, 3)
        self.downconv_7 = self.conv_down_block(512, 512, 3)

        self.upconv_7 = self.conv_up_block(512, 512)
        self.upconv_6 = self.conv_up_block(512, 512)
        self.upconv_5 = self.conv_up_block(512, 256)
        self.upconv_4 = self.conv_up_block(256, 128)
        self.upconv_3 = self.conv_up_block(128, 64)
        self.upconv_2 = self.conv_up_block(64, 32)
        self.upconv_1 = self.conv_up_block(32, 16)

        self.conv_7 = self.conv_block(1024, 512)
        self.conv_6 = self.conv_block(1024, 512)
        self.conv_5 = self.conv_block(512, 256)
        self.conv_4 = self.conv_block(256, 128)
        self.conv_3 = self.conv_block(130, 64)
        self.conv_2 = self.conv_block(66, 32)
        self.conv_1 = self.conv_block(18, 16)

        self.get_disp4 = self.disp_block(128)
        self.get_disp3 = self.disp_block(64)
        self.get_disp2 = self.disp_block(32)
        self.get_disp1 = self.disp_block(16)

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
            nn.Conv3d(1, out_dim, kernel_size=(in_dim, kernal, kernal), stride=1, padding=int((kernal - 1) / 2)),
            nn.BatchNorm3d(out_dim), nn.ELU()]  # h,w -> h,w
        # conv_down_block += [
        #     nn.Conv3d(out_dim, out_dim, kernel_size=(in_dim, kernal, kernal), stride=2, padding=int((kernal - 1) / 2)),
        #     nn.BatchNorm3d(out_dim), nn.ELU()]  # h,w -> h/2,w/2

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
        conv_1 = self.downconv_1(x)  # 32x128x256
        conv_1 = torch.squeeze(conv_1)
        print("CONV1 " * 10, conv_1.shape)
        conv_2 = self.downconv_2(conv_1)  # 64x64x128
        conv_3 = self.downconv_3(conv_2)  # 128x32x64
        conv_4 = self.downconv_4(conv_3)  # 256x16x32
        conv_5 = self.downconv_5(conv_4)  # 512x8x16
        conv_6 = self.downconv_6(conv_5)  # 512x4x8
        conv_7 = self.downconv_7(conv_6)  # 512x2x4

        conv7_up = self.upsample_(conv_7, 2)  # 512x4x8
        upconv_7 = self.upconv_7(conv7_up)  # 512x4x8

        concat_7 = torch.cat([upconv_7, conv_6], 1)  # 1024x4x8
        iconv_7 = self.conv_7(concat_7)  # 512x4x8

        iconv7_up = self.upsample_(iconv_7, 2)  # 512x8x16
        upconv_6 = self.upconv_6(iconv7_up)  # 512x8x16
        concat_6 = torch.cat([upconv_6, conv_5], 1)  # 1024x8x16
        iconv_6 = self.conv_6(concat_6)  # 512x8x16

        iconv6_up = self.upsample_(iconv_6, 2)  # 512x16x32
        upconv_5 = self.upconv_5(iconv6_up)  # 256x16x32
        concat_5 = torch.cat([upconv_5, conv_4], 1)  # 512x16x32
        iconv_5 = self.conv_5(concat_5)  # 256x16x32

        iconv5_up = self.upsample_(iconv_5, 2)  # 256x32x64
        upconv_4 = self.upconv_4(iconv5_up)  # 128x32x64
        concat_4 = torch.cat([upconv_4, conv_3], 1)  # 256x32x64
        iconv_4 = self.conv_4(concat_4)  # 128x32x64
        self.disp4 = 1.0 * self.get_disp4(iconv_4)  # 2x32x64
        udisp4 = self.upsample_(self.disp4, 2)  # 2x64x128

        iconv4_up = self.upsample_(iconv_4, 2)  # 128x64x128
        upconv_3 = self.upconv_3(iconv4_up)  # 64x64x128
        concat_3 = torch.cat([upconv_3, conv_2, udisp4], 1)  # 130x64x128
        # print("concat_3" * 10, concat_3.shape)
        iconv_3 = self.conv_3(concat_3)  # 64x64x128
        self.disp3 = 1.0 * self.get_disp3(iconv_3)  # 2x64x128
        udisp3 = self.upsample_(self.disp3, 2)  # 2x128x256

        iconv3_up = self.upsample_(iconv_3, 2)  # 64x128x256
        upconv_2 = self.upconv_2(iconv3_up)  # 32x128x256
        concat_2 = torch.cat([upconv_2, conv_1, udisp3], 1)  # 66x128x256
        iconv_2 = self.conv_2(concat_2)  # 32x128x256
        self.disp2 = 1.0 * self.get_disp2(iconv_2)  # 2x128x256
        udisp2 = self.upsample_(self.disp2, 2)  # 2x256x512

        iconv2_up = self.upsample_(iconv_2, 2)  # 32x256x512
        upconv_1 = self.upconv_1(iconv2_up)  # 16x256x512
        concat_1 = torch.cat([upconv_1, udisp2], 1)  # 18x256x512
        iconv_1 = self.conv_1(concat_1)  # 16x256x512
        self.disp1 = 1.0 * self.get_disp1(iconv_1)  # 2x256x512

        return [self.disp1, self.disp2, self.disp3, self.disp4]


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

            # rand_3color = 0.05 + np.random.rand(3)
            # rand_3color = rand_3color / np.sum(rand_3color)
            # R = rand_3color[0]
            # G = rand_3color[1]
            # B = rand_3color[2]

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

        stack = stack.astype(np.float32)
        stack = stack / 255.
        # print(stack.shape)
        return stack

    def loadDepth(self, folder):

        images = sorted(glob.glob(os.path.join(folder, "*.exr")))

        depth = cv2.imread(images[0], 2)

        depth = depth / np.max(depth)

        depth = np.expand_dims(depth, 0)
        return depth

    def __getitem__(self, idx):

        if idx not in self.cache:
            self.cache[idx] = {}
            self.cache[idx]['rgb'] = self.loadRGB(self.subfolders[idx])
            self.cache[idx]['depth'] = self.loadDepth(self.subfolders[idx])

        sample = {'rgb': self.cache[idx]['rgb'], 'depth': self.cache[idx]['depth']}

        return sample


net = mono_net(16, 2)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
net = net.to(device)

for param in net.parameters():
    param.requires_grad = True

lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)

# criterion = nn.MSELoss()

# criterion = nn.CrossEntropyLoss()


dataset = EpiDataset(folder='/tmp/gino/')
dataset_test = EpiDataset(folder='/tmp/gino/')

training_generator = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
validation_generator = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

for epoch in range(1000):
    # while True:
    print("EPOCH:", epoch)

    if epoch % 150 == 0 and epoch > 0:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("LEANING RAT CHANGED", "!" * 20)

    for batch in training_generator:

        # TRAINING
        net.train()

        input = batch['rgb']
        target = batch['depth']

        input = input.to(device)
        target = target.to(device)

        pyramid = net.scale_pyramid_(input, 4)
        target_pyramid = net.scale_pyramid_(target, 4)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            output = net(input)

            output = [torch.unsqueeze(d[:, 0, :, :], 1) for d in output]

            depth_loss = 0.0
            for index, out in enumerate(output):
                dl = torch.sum(torch.abs(output[index] - target_pyramid[index]))
                depth_loss += dl

            smoothness_x, smoothness_y = net.get_disparity_smoothness(output, pyramid)
            smoothness_loss = 0.0

            for index, s in enumerate(smoothness_x):
                smoothness_loss += (torch.mean(torch.abs(smoothness_x[index])) / (2 ** index))
                smoothness_loss += (torch.mean(torch.abs(smoothness_y[index])) / (2 ** index))

            loss = depth_loss + 0.0 * smoothness_loss

            print("Loss:", loss)

            loss.backward()
            optimizer.step()

    for batch in validation_generator:
        print("∞" * 20)
        print("TEST " * 20)
        net.eval()
        input = batch['rgb']
        target = batch['depth'].cpu().numpy()

        input = input.to(device)

        output = net(input)[0].detach().cpu().numpy()

        map_gt = (target[0][0] * 255).astype(np.uint8)
        map_pred = (output[0][0] * 255).astype(np.uint8)

        # print("SHAPE", map_pred.shape)
        cv2.imwrite("/tmp/gt.png", map_gt)
        cv2.imwrite("/tmp/pred.png", map_pred)
        break

    #
    # # d = dataset[2]
    # rgb = d['rgb']
    # depth = d['depth']
    #
    # rgb = np.expand_dims(rgb, 0)
    # depth = np.expand_dims(depth, 0)
    # depth = np.expand_dims(depth, 0)
    #
    # target = torch.Tensor(depth)
    # print("TARGET "*10, target.shape)
    #
    #
    #
    # input = torch.Tensor(rgb)
    # print(input.shape)
    #
    # output = net(input)
    # #
    # loss = torch.sqrt(criterion(output[0][:,0,:,:], target))
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    #
    # print("OUTPUT"*10, output[0].shape)
    #
    # slice = rgb[0,:, 100,:]
    # print(np.min(rgb), np.max(rgb), rgb.dtype)
    #
    # cv2.imshow("image", (slice*255).astype(np.uint8))
    # cv2.waitKey(0)
