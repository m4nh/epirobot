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

    def __init__(self, folder, is_test=False, is_negative=False):
        self.folder = folder
        self.images = sorted(glob.glob(os.path.join(self.folder, '*')))
        self.is_negative = is_negative

        if not is_test:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512, 512)),
                    # transforms.Grayscale(),
                    transforms.RandomAffine(180, (0.02, 0.02), fillcolor=9),
                    transforms.ToTensor()
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512, 512)),
                    # transforms.Grayscale(),
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


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=False,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class AnoNet(BaseNetwork):

    def __init__(self, name, checkpoints_path):
        super(AnoNet, self).__init__(name=name, checkpoints_path=checkpoints_path)

        self.input_channels = 1

        channels = 16
        self.conv1 = conv2DBatchNormRelu(self.input_channels, channels, 3, 1, 1)
        self.downsample1 = nn.MaxPool2d(2)

        channels *= 2
        self.conv2 = conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample2 = nn.MaxPool2d(2)

        channels *= 2
        self.conv3 = conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample3 = nn.MaxPool2d(2)

        channels *= 2
        self.conv4 = conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample4 = nn.MaxPool2d(2)

        channels *= 2
        self.conv5 = conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample5 = nn.MaxPool2d(2)

        channels *= 2
        self.conv6 = conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample6 = nn.MaxPool2d(2)

        channels *= 2
        self.conv7 = nn.Conv2d(int(channels / 2), channels, 3, 1,
                               1)  # conv2DBatchNormRelu(channels / 2, channels, 3, 1, 1)
        self.downsample7 = nn.MaxPool2d(2)

        channels /= 2
        self.upconv7 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv6 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv5 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv4 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv3 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv2 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv1 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        self.conv_last = nn.Conv2d(int(channels), self.input_channels, 3, 1, 1)
        self.prediction = nn.Tanh()

    # def buildLoss(self, output, target):
    #     loss = self.criterion(output, target)
    #     return loss

    def filterInputImage(self, x):
        return nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(x)
        x = self.conv2(x)
        x = self.downsample2(x)
        x = self.conv3(x)
        x = self.downsample3(x)
        x = self.conv4(x)
        x = self.downsample4(x)
        x = self.conv5(x)
        x = self.downsample5(x)
        x = self.conv6(x)
        x = self.downsample6(x)
        x = self.conv7(x)
        x = self.downsample7(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv7(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv6(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv5(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upconv1(x)

        x = self.conv_last(x)
        x = self.prediction(x)

        return (x + 1.0) / 2.0


model = AnoNet(name='anonet_l1ssim', checkpoints_path='/tmp')

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
model = model.to(device)
torchsummary.summary(model, (1, 512, 512))
for param in model.parameters():
    param.requires_grad = True

# OPTIMIZER
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

dataset = AnoDataset(folder='/tmp/ano_dataset_train')
dataset_neg = AnoDataset(folder='/tmp/ano_dataset_train_neg', is_negative=True)
dataset_test = AnoDataset(folder='/tmp/ano_dataset_test', is_test=True)

generator = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
generator_neg = DataLoader(dataset_neg, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
generator_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# LOAD MODEL IF ANY
model.loadModel()

LossL1 = nn.L1Loss()  # SSIM(11, reduction='mean')
LossSSIM = SSIM(5, reduction='mean')

for epoch in range(5000):

    print("EPOCH", epoch)

    if epoch % 10 == 0 and epoch > 0:
        model.saveModel()

    loss_ = 0.0
    counter = 0.0
    for gen in [generator]:  # , generator_neg]:
        for index, batch in enumerate(gen):
            model.train()
            optimizer.zero_grad()

            input = batch['input']
            input = input.to(device)

            target = batch['target']
            target = target.to(device)

            with torch.set_grad_enabled(True):
                output = model(input)

                input_r = input[:, 0, :, :]
                input_g = input[:, 1, :, :]
                input_b = input[:, 2, :, :]

                output_r = output[:, 0, :, :]
                output_g = output[:, 1, :, :]
                output_b = output[:, 2, :, :]

                loss1 = LossL1(input, output)

                loss2_r = LossSSIM(input_r, output_r)
                loss2_g = LossSSIM(input_g, output_g)
                loss2_b = LossSSIM(input_b, output_b)
                loss2 = 0.3 * loss2_b + 0.3 * loss2_g + 0.3 * loss2_r

                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                loss_ += loss.detach().cpu().numpy()
                counter += 1.0
                print("Batch: {}/{}".format(index, len(generator)))

    print("Loss", loss_ / counter)

    if True:
        stack = None
        max_stack = 20
        print("∞" * 20)
        print("TEST " * 20)
        for index, batch in enumerate(generator_test):

            model.eval()
            input = batch['input']
            input = input.to(device)

            output = model(input)
            output = output.detach()

            # print("TG", target[0].shape, np.min(target[0].cpu().numpy()), np.max(target[0].cpu().numpy()))
            # print("OPUT", output[0].shape, np.min(output[0].cpu().numpy()), np.max(output[0].cpu().numpy()))

            # print("INPUT ", input.shape)

            img_in = input[0]
            img_out = output[0]
            img_diff = torch.abs(img_in - img_out)

            rgb = AnoDataset.displayableImage(img_in)
            out = AnoDataset.displayableImage(img_out)
            diff = AnoDataset.displayableImage(img_diff)

            map = np.vstack((rgb, out, diff))

            if stack is None:
                stack = map
            else:
                stack = np.hstack((stack, map))

            index += 1
            if index >= max_stack:
                break

        cv2.imwrite("/tmp/ano_predictions.jpg", stack)

for d in generator:
    print(d.shape)
