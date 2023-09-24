import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
import collections


class ActCNN(nn.Module):
    def __init__(self, action_dim, deep, lent):
        super(ActCNN, self).__init__()
        self.lent = lent
        self.conv0 = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=2, padding=1)  # 224 -> 112
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=2, padding=1)  # 112 -> 56
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=2, padding=1)  # 56 -> 28
        # 448 -> 224 -> 112 -> 56 -> 28
        self.fc = nn.Linear(28 * 28 * 32, action_dim)

        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)

    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    def _gen_model(self, act):
        # 冻结参数的梯度
        feature_extract = True
        # 0.13版本后的新写法

        the_model = models.resnet18()
        save_dir = 'resnet18.pt'
        the_model.load_state_dict(torch.load(save_dir))
        the_model = self._set_parameter_requires_grad(
            the_model, feature_extract)
        # 修改模型
        num_ftrs = the_model.fc.in_features
        the_model.fc = nn.Linear(in_features=num_ftrs,
                                 out_features=act, bias=True)

        return the_model

    def forward(self, x):
        if self.lent > 224:
            x = F.relu(self.bn0(self.conv0(x)))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        x = (x - x.max())  # maybe its 0 /max
        x = F.softmax(x, dim=1)
        return x


class ValueCNN(nn.Module):
    def __init__(self, action_dim, deep, lent):
        super(ValueCNN, self).__init__()
        self.lent = lent
        self.conv0 = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=2, padding=1)  # 224 -> 112
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=2, padding=1)  # 112 -> 56
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=2, padding=1)  # 56 -> 28
        # 448 -> 224 -> 112 -> 56 -> 28
        self.fc = nn.Linear(28 * 28 * 32, action_dim)

        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)

    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    def _gen_model(self, act):
        # 冻结参数的梯度
        feature_extract = True
        # 0.13版本后的新写法

        the_model = models.resnet18()
        save_dir = 'resnet18.pt'
        the_model.load_state_dict(torch.load(save_dir))
        the_model = self._set_parameter_requires_grad(
            the_model, feature_extract)
        # 修改模型
        num_ftrs = the_model.fc.in_features
        the_model.fc = nn.Linear(in_features=num_ftrs,
                                 out_features=act, bias=True)

        return the_model

    def forward(self, x):
        if self.lent > 224:
            x = F.relu(self.bn0(self.conv0(x)))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))
        #x = (x - x.max())  # maybe its 0 /max
        # x = F.softmax(x, dim=1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 in_channel,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(in_channel, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = (x - x.max())  # maybe its 0 /max
            x = F.softmax(x, dim=1)

        return x


def resnet34(in_channel, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(in_channel, BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(in_channel, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(in_channel, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(in_channel, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(in_channel, Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(in_channel, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(in_channel, Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(in_channel, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(in_channel, Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
