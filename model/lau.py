import torch
import torch.nn as nn
from model import common
import math
from spherenet.sphere_cnn import SphereConv2D
from textureTransformer.non_local_dot_product import NONLocalBlock2D


def make_model(opt):
    return LAUNet(opt)


class Evaluator(nn.Module):
    def __init__(self, n_feats):
        super(Evaluator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, n_feats, kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer = nn.Conv2d(in_channels=n_feats, out_channels=2, kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        # saved actions and rewards
        self.saved_action = None
        self.rewards = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        if self.training:
            m = torch.distributions.Categorical(softmax)
            action = m.sample()
            '''action = softmax.multinomial(num_samples=1)'''
            self.saved_action = action
        else:
            action = softmax[1]
            action = torch.where(action > 0.5, 1, 0)
            self.saved_action = action
            m = None
        return action, m


class LAUNet(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(LAUNet, self).__init__()
        self.opt = opt
        self.scale = opt.scale[-1]
        self.level = int(math.log(self.scale, 2))
        self.saved_actions = []
        self.softmaxs = []
        n_blocks = opt.n_blocks
        n_feats = 64
        kernel_size = 3
        n_height = 1024

        # main SR network
        self.upsample = [nn.Upsample(scale_factor=2**(i+1), mode='bicubic', align_corners=False) for i in range(self.level)]
        self.upsample = nn.ModuleList(self.upsample)

        rgb_mean = (0.4737, 0.4397, 0.4043)
        rgb_std = (1.0, 1.0, 1.0)
        
        # data preprocessing
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)
        # head conv
        self.head = conv(opt.n_colors, n_feats)
        # CA Dense net
        self.body = [common.CADensenet(SphereConv2D, n_feats, n_CADenseBlocks=n_blocks) for _ in range(self.level+1)]
        self.body = nn.ModuleList(self.body)
        # Texture transformer (tt)
        self.tt = [NONLocalBlock2D(n_feats) for _ in range(self.level)]
        self.tt = nn.ModuleList(self.body)
        # bypass network
        self.bypass = [common.Resnet(conv, n_feats, kernel_size=3, num=5, depth=5) for _ in range(self.level)]
        self.bypass = nn.ModuleList(self.bypass)
        # upsample blocks
        self.up_blocks = [common.Upsampler(common.default_conv, 2, n_feats, act=False) for _ in
                          range(2 * self.level - 1)]
        self.up_blocks += [common.Upsampler(common.default_conv, 2 ** i, 3, act=False) for i in
                           range(self.level, 0, -1)]
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # tail conv that output sr ODIs
        self.tail = [conv(n_feats, opt.n_colors) for _ in range(self.level+1)]
        self.tail = nn.ModuleList(self.tail)
        # data postprocessing
        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)
        # evaluator subnet
        self.evaluator = nn.ModuleList()
        for p in range(opt.n_evaluator):
            self.evaluator.append(Evaluator(n_feats))

    def merge(self, imglist, radio):
        if radio[0] == 0 and radio[-1] == 0:
            return imglist[-1]
        else:
            result = [imglist[0]]
            for i in range(1, len(imglist)):
                north, middle, south = torch.split(result[-1],
                                                   [radio[0] * i, result[-1].size(2) - radio[0] * i - radio[-1] * i,
                                                    radio[-1] * i], dim=2)
                result.append(torch.cat((north, imglist[i], south), dim=2))
            return result[-1]

    def forward(self, lr):
        masks = []
        gprobs = []
        x = self.sub_mean(lr)
        g1 = x
        g2 = self.upsample[0](x)
        g3 = self.upsample[1](x)
        input = self.head(x)
        # 1st level
        b1 = self.body[0](input)
        b1 = self.tt[0](b1)
        f1 = self.tail[0](b1)
        g1 = self.add_mean(f1+ g1)

        eva_g1 = g1.detach()
        patchlist = torch.chunk(eva_g1, self.opt.n_evaluator, dim=2)
        for i in range(len(patchlist)):
            action, gprob = self.evaluator[i](patchlist[i])
            threshold = int(action.size(0))+1 if self.training else 1
            mask = 1 if int(action.sum()) == threshold else 0
            self.saved_actions.append(action)
            self.softmaxs.append(gprob)
            masks.append(mask)
            gprobs.append(gprob)
        crop_n, remain, crop_s = 0, 0, 0
        for i in range(self.opt.n_evaluator//(2**self.level)):
            if masks[i] == 1:
                crop_n += b1.size(2)//self.opt.n_evaluator
            else:
                break
        for j in range(self.opt.n_evaluator-1, self.opt.n_evaluator*((2**self.level-1)//(2**self.level)), -1):
            if masks[j] == 1:
                crop_s += b1.size(2)//self.opt.n_evaluator
            else:
                break
        remain = g1.size(2)-crop_n-crop_s
        if crop_n or crop_s:
            _, b1, _ = torch.split(b1, [crop_n, remain, crop_s], dim=2)
            _, g2, _ = torch.split(g2, [crop_n*2, remain*2, crop_s*2], dim=2)
        g1 = self.bypass[0](g1)

        # 2ed level
        b2 = self.up_blocks[0](b1)
        b2 = self.body[1](b2)
        b2 = self.tt[1](b2)
        f2 = self.tail[1](b2)
        g2 = self.add_mean(f2 + g2)
        if crop_n or crop_s:
            _, b2, _ = torch.split(b2, [crop_n * 2, b2.size(2) - crop_n * 2 - crop_s * 2, crop_s * 2], dim=2)
            _, g3, _ = torch.split(g3, [crop_n * 8, g3.size(2) - crop_n * 8 - crop_s * 8, crop_s * 8], dim=2)
        g2 = self.bypass[1](g2)
        # 3rd level
        b3 = self.up_blocks[1](b2)
        b3 = self.body[2](b3)
        b3 = self.tt[2](b3)
        f3 = self.tail[2](b3)
        g3 = self.add_mean(f3 + g3)

        g1up = self.up_blocks[3](g1)
        g2up = self.up_blocks[4](g2)
        out = self.merge([g1up, g2up, g3], [crop_n * 4, remain * 4, crop_s * 4])
        results = [g1up, g2up, g3, out]

        return results
