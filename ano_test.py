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

class AnoDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder
        self.images = sorted(glob.glob(os.path.join(self.folder,'*')))

    def __len__(self):
        return len(self.images)


    # def getImageTransformPipeline(self, brightness=0.5, contrast=0.3, saturation=0.2, hue=0.3):
    #     # brightness = 0.3
    #     # contrast = 0.3
    #     # saturation = 1.0
    #     # hue = 0.5
    #
    #     trans = []
    #     if brightness > 0:
    #         brightness_factor = np.random.uniform(max(0.0, 1 - brightness), 1 + brightness)
    #         trans.append(Lambda(lambda img: FT.adjust_brightness(img, brightness_factor)))
    #
    #     if contrast > 0:
    #         contrast_factor = np.random.uniform(max(0.0, 1 - contrast), 1 + contrast)
    #         trans.append(Lambda(lambda img: FT.adjust_contrast(img, contrast_factor)))
    #
    #     if saturation > 0:
    #         saturation_factor = np.random.uniform(max(0.0, 1 - saturation), 1 + saturation)
    #         trans.append(Lambda(lambda img: FT.adjust_saturation(img, saturation_factor)))
    #
    #     if hue > 0:
    #         hue_factor = np.random.uniform(-hue, hue)
    #         trans.append(Lambda(lambda img: FT.adjust_hue(img, hue_factor)))
    #
    #     random.shuffle(trans)
    #     trans = [transforms.ToPILImage()] + trans + [transforms.ToTensor()]
    #     trans = transforms.Compose(trans)
    #     return trans

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
        return img


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
            with_bn=True,
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

        channels = 32
        self.conv1 = conv2DBatchNormRelu(3, channels, 3, 1, 1)
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

        channels /= 2
        self.upconv4 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv3 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv2 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        channels /= 2
        self.upconv1 = conv2DBatchNormRelu(channels * 2, channels, 3, 1, 1)

        self.prediction = nn.Conv2d(int(channels), 3, 3, 1, 1)


    # def buildLoss(self, output, target):
    #     loss = self.criterion(output, target)
    #     return loss

    def filterInputImage(self,x):
        return nn.functional.interpolate(x, size=(512,512), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.filterInputImage(x)
        x = self.conv1(x)
        x = self.downsample1(x)
        x = self.conv2(x)
        x = self.downsample2(x)
        x = self.conv3(x)
        x = self.downsample3(x)
        x = self.conv4(x)
        x = self.downsample4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upconv4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upconv3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upconv2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upconv1(x)

        x = self.prediction(x)

        return x


model = AnoNet(name='anonet', checkpoints_path='/tmp')
# torchsummary.summary(model, (3, 700, 700))

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

# OPTIMIZER
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)


dataset = AnoDataset(folder='/tmp/ano_dataset_train')
dataset_test = AnoDataset(folder='/tmp/ano_dataset_test')
generator = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
generator_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


# LOAD MODEL IF ANY
model.loadModel()


criterion = nn.MSELoss()


for epoch in range(5000):

    print("EPOCH", epoch)

    if epoch % 10 == 0 and epoch > 0:
        model.saveModel()

    loss_ = 0.0
    counter = 0.0
    for index, batch in enumerate(generator):
        model.train()
        optimizer.zero_grad()

        input = batch
        input = input.to(device)

        with torch.set_grad_enabled(True):

            output = model(input)
            loss = criterion(nn.functional.interpolate(input, size=(512,512), mode='bilinear', align_corners=True), output)

            loss.backward()
            optimizer.step()

            loss_ += loss.detach().cpu().numpy()
            counter += 1.0
            print("Batch: {}/{}".format(index, len(generator)))

    print("Loss", loss_/counter)

    if True:
        stack = None
        max_stack = 8
        print("∞" * 20)
        print("TEST " * 20)
        for index, batch in enumerate(generator_test):

            model.eval()
            input = batch
            input = input.to(device)

            output = model(input)
            output = output.detach()

            # print("TG", target[0].shape, np.min(target[0].cpu().numpy()), np.max(target[0].cpu().numpy()))
            # print("OPUT", output[0].shape, np.min(output[0].cpu().numpy()), np.max(output[0].cpu().numpy()))


            # print("INPUT ", input.shape)
            rgb = AnoDataset.displayableImage(model.filterInputImage(input)[0])
            out = AnoDataset.displayableImage(model.filterInputImage(output)[0])

            map = np.vstack((rgb, out))

            if stack is None:
                stack = map
            else:
                stack = np.hstack((stack, map))

            index += 1
            if index >= max_stack:
                break

        cv2.imwrite("/tmp/ano_predictions.png", stack)

for d in generator:
    print(d.shape)