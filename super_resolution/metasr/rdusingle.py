from model import common
import functools
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def make_model(args, parent=False):
    model = MetaRDU(args)
    #model.apply(init_weights)
    return model

class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 9, padding=4)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        #return self.actv_2(self.conv_2(x))
        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x

class MetaRDU(nn.Module):
    def __init__(self, args):
        super(MetaRDU, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        conv=common.default_conv
        n_feats = 128
        kernel_size = 3
        in_nc = self.args.n_colors

        filters_0 = 128
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0
        # Encoder:
        # Level 0:
        self.input_block = InputBlock(in_nc, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        #self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        #self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        #self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        #self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        #self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        #self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        #self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, in_nc)
        ## position to weight
        self.upsampler = common.Upsampler(conv, self.args.scale2[0], n_feats, act=False) 
                                 
        self.tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*self.tail)
        ##

    def forward(self, x, pos_mat, a):
        inputs = x
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
 
        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
    
        out_4 = self.up_2([out_3, out_2])   # Level 2, 32
        out_4 = self.block_2_2(out_4)
       
        out_5 = self.up_1([out_4, out_1])   # Level 1, 64
        out_5 = self.block_1_2(out_5)
       
        out_6 = self.up_0([out_5, out_0])   # Level 0, 128
        out_6 = self.block_0_2(out_6)
        #print(out_6.shape)
        #########################################
        out = self.upsampler(out_6)
        out = self.tail(out)

        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        #self.scale = self.args.scale[scale_idx]
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
        #print(self.scale)


