import torch
import torch.nn as nn
from torch.distributions.binomial import Binomial
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class Net(nn.Module):
    """
    A Residual network.
    """
    #  Dropout totally be controlled by random, not by model.train()

    def __init__(self, retrain=False, random=False):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res1 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res2 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features=1024, out_features=10, bias=True)
        
        self.dropout = nn.Dropout(p=0.2)
        self.retrain= retrain
        self.random = random

        self.penultimate = None
        self.previous = None
        self.mask1 = None
        self.mask2 = None


    def forward(self, x, mask_update=False):
        out = self.conv1(x)
        out = self.res1(out)
        out = self.conv2(out)

        if self.random == True:
            # out = self.dropout(out)
            # out = F.dropout(out, p=0.2, training=True)
            if mask_update == True:
                self.mask1 = Binomial(probs=0.5).sample(out.size()).to(out.device)/0.5
            if self.mask1 != None:
                out = out * self.mask1

        out = self.res2(out)
        self.previous = out

        out = self.maxpool(out)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])

        self.penultimate = out

        # if self.random == True or self.retrain == True:
        if self.random == True and mask_update == True:
            # out = self.dropout(out)
            # out = F.dropout(out, p=0.2, training=True)
            if mask_update == True:
                self.mask2 = Binomial(probs=0.5).sample(out.size()).to(out.device)/0.5
            if self.mask2 != None:
                out = out * self.mask2

        out = self.linear(out)
        return out
