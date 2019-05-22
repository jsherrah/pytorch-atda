"""Feature encoder for ATDA.

it's called as `shared network` in the paper.
"""

import torch
from torch import nn
import torchvision.models as models


class EncoderVGG(nn.Module):
    """Use VGG as encoder"""

    def __init__(self):
        """Init encoder."""
        super(EncoderVGG, self).__init__()
        self.restored = False
        self.encoder = models.__dict__['vgg16'](pretrained=True).features

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        return out.view(out.shape[0], out.shape[1:].numel())#768)

class EncoderA(nn.Module):
    """Feature encoder class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self):
        """Init encoder."""
        super(EncoderA, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [3 x 28 x 28]
            # output [32 x 12 x 12]
            nn.Conv2d(3, 32, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2nd conv block
            # input [32 x 12 x 12]
            # output [48 x 4 x 4]
            nn.Conv2d(32, 48, 5, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        return out.view(out.shape[0], out.shape[1:].numel())#768)


class EncoderB(nn.Module):
    """Feature encoder class for MNIST -> SVHN in ATDA."""

    def __init__(self):
        """Init encoder."""
        super(EncoderB, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [1 x 28 x 28]
            # output [64 x 13 x 13]
            nn.Conv2d(3, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 2nd conv block
            # input [64 x 13 x 13]
            # output [64 x 6 x 6]
            nn.Conv2d(64, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3rd conv block
            # input [64 x 6 x 6]
            # output [128 x 6 x 6]
            nn.Conv2d(64, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # FC block
            nn.Linear(4608, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        return out


class EncoderC(nn.Module):
    """Feature encoder class for SVHN -> MNIST or SYN Digits -> SVHN."""

    def __init__(self):
        """Init encoder."""
        super(EncoderC, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [3 x 32 x 32]
            # output [64 x 15 x 15]
            nn.Conv2d(3, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 2nd conv block
            # input [64 x 15 x 15]
            # output [64 x 7 x 7]
            nn.Conv2d(64, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3rd conv block
            # input [64 x 7 x 7]
            # output [128 x 7 x 7]
            nn.Conv2d(64, 128, 5, 1, 2, bias=False),
            nn.ReLU(),
            # FC block
            nn.Linear(6272, 3072),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward encoder."""
        out = self.encoder(x)
        return out


# class EncoderD(nn.Module):
#     """Feature encoder class for SYN Signs -> GTSRB in ATDA."""
#
#     def __init__(self):
#         """Init encoder."""
#         super(EncoderD, self).__init__()
#
#         self.restored = False
#
#         self.encoder = nn.Sequential(
#             # 1st conv block
#             # input [3 x 32 x 32]
#             # output [64 x 15 x 15]
#             nn.Conv2d(3, 96, 5, 1, 0, bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # 2nd conv block
#             # input [64 x 15 x 15]
#             # output [64 x 7 x 7]
#             nn.Conv2d(96, 144, 5, 1, 0, bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # 3rd conv block
#             # input [64 x 7 x 7]
#             # output [128 x 7 x 7]
#             nn.Conv2d(144, 256, 5, 1, 0, bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#
#     def forward(self, x):
#         """Forward encoder."""
#         out = self.encoder(x)
#         return out.view(-1, 1)
