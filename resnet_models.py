from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import os


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class conv3D(nn.Module):
    def __init__(self, num_in_depth, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv3D, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=(num_in_depth,kernel_size, kernel_size),
                                   stride=stride)
        self.normalize = nn.BatchNorm3d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4 * num_out_layers)

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=1):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return self.sigmoid(x)


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResnetModel(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=1, encoder='resnet18', pretrained=False):
        super(ResnetModel, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', \
                           'resnet101', 'resnet152'], \
            "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder) \
            (pretrained=pretrained)
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                                       kernel_size=(7, 7), stride=(2, 2),
                                       padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.encoder1 = resnet.layer1  # H/4
        self.encoder2 = resnet.layer2  # H/8
        self.encoder3 = resnet.layer3  # H/16
        self.encoder4 = resnet.layer4  # H/32

        # decoder
        self.upconv6 = upconv(filters[3], 512, 3, 2)
        self.iconv6 = conv(filters[2] + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(filters[0] + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128, num_out_layers)

        self.upconv3 = upconv(128, 64, 3, 1)  #
        self.iconv3 = conv(64 + 64 + num_out_layers, 64, 3, 1)
        self.disp3_layer = get_disp(64, num_out_layers)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + num_out_layers, 32, 3, 1)
        self.disp2_layer = get_disp(32, num_out_layers)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + num_out_layers, 16, 3, 1)
        self.disp1_layer = get_disp(16, num_out_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def buildLoss(self, target):
        pass

    def forward(self, x):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        # skips
        skip1 = x_first_conv
        skip2 = x_pool1
        skip3 = x1
        skip4 = x2
        skip5 = x3

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4


class EpiFeatures(nn.Module):
    def __init__(self, depth, num_in_layers, num_out_layers, kernel_size=3, num_blocks=3):
        super(EpiFeatures, self).__init__()

        self.conv_7x7 = conv(num_in_layers, num_out_layers, 7, 1)
        self.conv_5x5 = conv(num_in_layers, num_out_layers, 5, 1)
        self.conv_3x3 = conv(num_in_layers, num_out_layers, 3, 1)

        self.out_size = num_out_layers * 3 * depth
        # self.features = self.build(num_in_layers, num_out_layers, kernel_size, num_blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # print(x.shape)

        outs = []
        for l in range(x.shape[2]):
            slice = x[:, :, l, :, :]
            f7 = self.conv_7x7(slice)
            f5 = self.conv_5x5(slice)
            f3 = self.conv_3x3(slice)
            f = torch.cat((f7, f5, f3), 1)
            # print(slice.shape, f.shape)

            outs += [f]

        out = torch.cat(outs, 1)

        # print(out.shape)
        return out


class BaseNetwork(nn.Module):

    def __init__(self, name, checkpoints_path):
        super(BaseNetwork, self).__init__()
        self.name = name
        self.checkpoints_path = checkpoints_path
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")


    def loadModel(self, tag='LAST'):
        last_model_path = os.path.join(self.checkpoints_path, "{}_{}.pb".format(self.name, tag))
        if os.path.exists(last_model_path):
            self.load_state_dict(torch.load(last_model_path))
            print("*" * 10, self.name, "MODEL LOADED!", "*" * 10)

    def saveModel(self, tag='LAST'):

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        torch.save(self.state_dict(), os.path.join(self.checkpoints_path, "{}_{}.pb".format(self.name, tag)))


class EpiNet(BaseNetwork):
    def __init__(self, num_in_layers, num_out_layers, name, checkpoints_path):
        super(EpiNet, self).__init__(name=name, checkpoints_path=checkpoints_path)

        self.epifeatures = EpiFeatures(11, 3, 8)
        self.resnet = ResnetModel(self.epifeatures.out_size, num_out_layers)
        self.criterion = torch.nn.L1Loss()

    def buildLoss(self, output, target):
        loss = self.criterion(output, target)
        return loss

    def forward(self, x):
        f = self.epifeatures(x)
        return self.resnet(f)


class EpinetSimple(BaseNetwork):  # vgg version
    def __init__(self, depth, input_nc, output_nc, name, checkpoints_path):
        super(EpinetSimple, self).__init__(name, checkpoints_path)

        self.features_layer = conv3D(11, 3, 256, 3, 1)  # EpiFeatures(depth, input_nc, 16)
        self.l1 = resblock(256, 128, 10, 1)
        self.out = get_disp(128 * 4, output_nc)
        self.criterion = torch.nn.L1Loss()

    def buildLoss(self, output, target):
        loss = self.criterion(output, target)
        return loss

    def forward(self, x):
        # 3x256x512
        debug = True

        l0 = self.features_layer(x)
        l0 = torch.squeeze(l0, 2)
        l1 = self.l1(l0)
        o = self.out(l1)
        return o
