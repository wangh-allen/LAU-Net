import math
import numpy as np
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DenseBlock(nn.Module):
    def __init__(self, depth=8, rate=8, input_dim=64, out_dims=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        filters = out_dims - rate * depth
        self.dense_module = [
            nn.Sequential(
                nn.Conv2d(input_dim, filters+rate, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        ]

        for i in range(1, depth):
            self.dense_module.append(
                 nn.Sequential(
                    nn.Conv2d(filters+i*rate, rate, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                 )
            )
        self.dense_module = nn.ModuleList(self.dense_module)

    def forward(self, x):
        features = [x]
        x = self.dense_module[0](features[-1])
        features.append(x)
        for idx in range(1, self.depth):
            x = self.dense_module[idx](features[-1])
            features.append(x)
            features[-1] = torch.cat(features[-2:], 1)
        return features[-1]


class CADensenet(nn.Module):
    def __init__(self, conv, n_feat, n_CADenseBlocks=5):
        super(CADensenet, self).__init__()
        self.n_blocks = n_CADenseBlocks

        denseblock = [
            DenseBlock(input_dim=n_feat, out_dims=64) for _ in range(n_CADenseBlocks)]
        calayer = []
        # The rest upsample blocks
        for _ in range(n_CADenseBlocks):
            calayer.append(CALayer(n_feat, reduction=16))

        self.CADenseblock = nn.ModuleList()
        for idx in range(n_CADenseBlocks):
            self.CADenseblock.append(nn.Sequential(denseblock[idx], calayer[idx]))
        self.CADenseblock.append(nn.Conv2d((n_CADenseBlocks+1)*n_feat, n_feat, kernel_size=1))

    def forward(self, x):
        feat = [x]
        for idx in range(self.n_blocks):
            x = self.CADenseblock[idx](feat[-1])
            feat.append(x)
        x = torch.cat(feat[:], 1)
        x = self.CADenseblock[-1](x)
        return x


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res