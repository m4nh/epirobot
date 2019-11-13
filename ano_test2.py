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
from torch.utils.tensorboard import SummaryWriter
import torchvision
from anomaleye import ElasticAE


class AnoDataset(Dataset):

    def __init__(self, folder, is_test=False, is_negative=False, resize=256):
        self.folder = folder
        self.images = sorted(glob.glob(os.path.join(self.folder, '*')))
        self.is_negative = is_negative

        if not is_test:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((resize, resize)),
                    # transforms.Grayscale(),
                    # transforms.RandomAffine(180, (0.02, 0.02), fillcolor=9),
                    transforms.ToTensor()
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((resize, resize)),
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
                                          nn.LeakyReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.LeakyReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class AnoEncoder(nn.Module):

    def __init__(self, input_channels):
        super(AnoEncoder, self).__init__()
        self.input_channels = input_channels

        channels = 32
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

    def forward(self, x, full_output=False):
        x = self.conv1(x)
        l1 = x

        x = self.downsample1(x)
        x = self.conv2(x)
        l2 = x
        x = self.downsample2(x)
        x = self.conv3(x)
        l3 = x
        x = self.downsample3(x)
        x = self.conv4(x)
        l4 = x
        x = self.downsample4(x)
        x = self.conv5(x)
        l5 = x
        x = self.downsample5(x)
        x = self.conv6(x)
        l6 = x
        x = self.downsample6(x)
        x = self.conv7(x)
        l7 = x
        x = self.downsample7(x)
        return l1, l2, l3, l4, l5, l6, l7, x


class AnoNet(BaseNetwork):

    def __init__(self, name, input_channels, checkpoints_path):
        super(AnoNet, self).__init__(name=name, checkpoints_path=checkpoints_path)

        self.input_channels = input_channels

        channels = 2048

        self.encoder = AnoEncoder(input_channels)

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

    def forward(self, x):
        l1, l2, l3, l4, l5, l6, l7, x = self.encoder(x)

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

        return l1, l2, l3, l4, l5, l6, l7, (x + 1.0) / 2.0


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# enc = AnoEncoder(3)
# torchsummary.summary(enc, (3, 512, 512))
# import sys
# sys.exit(0)

image_resize = 256
input_channels = 3
model = ElasticAE('elasticae', image_resize, input_channels=input_channels, output_channels=input_channels, latent_size=500,
                  layers=4,
                  initial_filters=16, checkpoints_path='/tmp/anomaleye/')

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
model = model.to(device)
torchsummary.summary(model, (input_channels, image_resize, image_resize))
for param in model.parameters():
    param.requires_grad = True

# tensorboard
writer = SummaryWriter("/tmp/anomaleye/runs")

# OPTIMIZER
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

dataset = AnoDataset(folder='/tmp/anomaleye/dataset/train', resize=image_resize)
# dataset_neg = AnoDataset(folder='/tmp/ano_dataset_train_neg', is_negative=True)
dataset_test = AnoDataset(folder='/tmp/anomaleye/dataset/test', is_test=True, resize=image_resize)

generator = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=False)
# generator_neg = DataLoader(dataset_neg, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
generator_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

# LOAD MODEL IF ANY
model.loadModel()

LossL1 = nn.L1Loss()  # SSIM(11, reduction='mean')
FeaturesLoss = nn.L1Loss()
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

                # input_r = torch.unsqueeze(input[:, 0, :, :], 1)
                # input_g = torch.unsqueeze(input[:, 1, :, :], 1)
                # input_b = torch.unsqueeze(input[:, 2, :, :], 1)
                #
                # output_r = torch.unsqueeze(output[:, 0, :, :], 1)
                # output_g = torch.unsqueeze(output[:, 1, :, :], 1)
                # output_b = torch.unsqueeze(output[:, 2, :, :], 1)

                loss1 = LossL1(input, output)

                # loss2_r = LossSSIM(input_r, output_r)
                # loss2_g = LossSSIM(input_g, output_g)
                # loss2_b = LossSSIM(input_b, output_b)
                # loss2 = 0.3 * loss2_b + 0.3 * loss2_g + 0.3 * loss2_r

                loss = loss1  # + loss2 + loss3

                if index == len(gen) - 1:
                    writer.add_scalar('Loss/reconstruction', loss1, epoch)
                    # writer.add_scalar('Loss/ssim', loss2, epoch)
                    # writer.add_scalar('Loss/features', loss3, epoch)
                    writer.add_image('Train/input_images', torchvision.utils.make_grid(input), epoch)
                    writer.add_image('Train/reconstructed_images', torchvision.utils.make_grid(output), epoch)

                loss.backward()
                optimizer.step()

                print("Batch: {}/{}".format(index, len(generator)))

    if True:
        print("∞" * 20)
        print("TEST " * 20)
        for index, batch in enumerate(generator_test):
            model.eval()
            input = batch['input']
            input = input.to(device)

            output = model(input).detach()

            writer.add_image('Test/input_images', torchvision.utils.make_grid(input), epoch)
            writer.add_image('Test/reconstructed_images', torchvision.utils.make_grid(output), epoch)
            break

writer.close()