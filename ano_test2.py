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
import argparse

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
                    transforms.RandomAffine(5, (0.05, 0.05), fillcolor=9),
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


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)



parser = argparse.ArgumentParser()
parser.add_argument("--name", default="anomaleye_net", type=str)
args = parser.parse_args()

image_resize = 256
input_channels = 3
model = ElasticAE(args.name, image_resize, input_channels=input_channels, output_channels=input_channels, latent_size=500,
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

# for d in dataset:
#     img = d['input']
#     img = dataset.displayableImage(img)
#     cv2.imshow("image",img)
#     cv2.waitKey(0)
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

                if index == len(gen) - 1 and epoch % 10 == 0:
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
