## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

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
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()

        self.args = args
        self.scale_idx = 0

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)
        conv=common.default_conv
        # RGB mean for DIV2K
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module

        #self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        self.upsampler = common.Upsampler(conv, self.args.scale2[0], n_feats, act=False) 
                                 
        self.tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*self.tail)

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat, a):
        #x = self.sub_mean(x)
        inputs = x
        out_1 = self.head(inputs)
        out_2 = self.body(out_1)
        #########################################
        out = self.upsampler(out_2)
        out = self.tail(out)
        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
