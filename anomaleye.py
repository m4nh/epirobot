import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from resnet_models import BaseNetwork


class Conv2DA(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_activation=True,
                 with_bn=False):
        super(Conv2DA, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        self.layer = nn.Sequential(conv_mod)

        if with_activation:
            self.layer.add_module("activation", nn.LeakyReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs


class TConv2DA(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, outpadding, bias=True, dilation=1,
                 with_activation=True,
                 with_bn=False):
        super(TConv2DA, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                                      int(n_filters),
                                      kernel_size=k_size,
                                      padding=padding,
                                      output_padding=outpadding,
                                      stride=stride,
                                      bias=bias,
                                      dilation=dilation, )

        self.layer = nn.Sequential(conv_mod)

        if with_activation:
            self.layer.add_module("activation", nn.LeakyReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), self.shape[0], self.shape[1], self.shape[2])


class ElasticEncoder(nn.Module):

    def __init__(self, input_size, input_channels, latent_size=1000, n_layers=4, initial_filters=16,
                 layers_mult=2):
        super(ElasticEncoder, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        self.layers = nn.Sequential()

        start_filters = input_channels
        last_filters = initial_filters * layers_mult
        for i in range(n_layers):
            self.layers.add_module("layer_{}".format(i), Conv2DA(start_filters, last_filters, 3, 2, 1))
            start_filters = last_filters
            last_filters = start_filters * layers_mult

        final_size = int(input_size / (2 ** n_layers))
        # self.layers.add_module("pre_features", Conv2DA(start_filters, start_filters, 3, 1, 1, with_activation=False))

        self.layers.add_module("flatten", Flatten())
        self.layers.add_module("features", nn.Linear(final_size * final_size * start_filters, latent_size))

    def forward(self, x):
        return self.layers(x)


class ElasticDecoder(nn.Module):

    def __init__(self, output_size, output_channels, latent_size=1000, n_layers=4, initial_filters=256,
                 layers_mult=2):
        super(ElasticDecoder, self).__init__()
        self.output_size = output_size

        final_size = int(output_size / (2 ** n_layers))

        self.layers = nn.Sequential()

        initial_size = initial_filters * final_size * final_size
        self.layers.add_module("Features", nn.Linear(latent_size, initial_size))
        self.layers.add_module("UnFlatten", UnFlatten([initial_filters, final_size, final_size]))

        start_filters = initial_filters
        last_filters = int(initial_filters / layers_mult)
        for i in range(n_layers):
            self.layers.add_module("layer_{}".format(i), TConv2DA(start_filters, last_filters, 3, 2, 1, 1))
            start_filters = last_filters
            last_filters = int(start_filters / layers_mult)
        #
        # final_size = int(input_size / (2 ** n_layers))
        # # self.layers.add_module("pre_features", Conv2DA(start_filters, start_filters, 3, 1, 1, with_activation=False))
        #
        self.layers.add_module("last_layer", Conv2DA(start_filters, output_channels, 3, 1, 1))
        self.layers.add_module("activation", nn.Sigmoid())
        # self.layers.add_module("features", nn.Linear(final_size * final_size * start_filters, latent_size))

    def forward(self, x):
        return self.layers(x)


class ElasticAE(BaseNetwork):

    def __init__(self, name, input_size, input_channels, output_channels, latent_size=1000, layers=4,
                 initial_filters=16, checkpoints_path='/tmp'):
        super(ElasticAE, self).__init__(name, checkpoints_path)

        multiplier = 2
        self.encoder = ElasticEncoder(input_size, input_channels, latent_size, layers, initial_filters, multiplier)
        self.decoder = ElasticDecoder(input_size, output_channels, latent_size, layers,
                                      initial_filters * (multiplier ** layers), multiplier)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# model = ElasticEncoder(256, 3)
# torchsummary.summary(model, (3, 256, 256))

# model = ElasticAE("miao",256, 3, 3, 500, 4, 16)
# torchsummary.summary(model, (3, 256, 256))
