from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_model(args, parent=False):
    return MetaMSRN(args)

class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output

class MetaMSRN(nn.Module):
    def __init__(self, args):
        super(MetaMSRN, self).__init__()
        
        n_feats = 64
        n_blocks = 8
        kernel_size = 9
        act = nn.ReLU(True)
        conv=common.default_conv

        self.scale = 1
        self.args = args
        self.scale_idx = 0
        self.n_blocks = n_blocks
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB(n_feats=n_feats))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        #self.upsampler = common.Upsampler(conv, self.args.scale2[0], n_feats, act=False) 
        self.upsampler = common.Upsampler(conv, self.args.scaleForTest2, n_feats, act=False) 
                                 
        self.tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*self.tail)

        self.fusion = nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1)
       
        
    def forward(self, x, pos_mat, a):
        x = self.head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1)
        res = self.fusion(res)
        #########################################
        out = self.upsampler(res)
        out = self.tail(out)
        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
