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
    model = MetaRRDB(args)
    #model.apply(init_weights)
    return model

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()

        self.layersRDB = []
        for _ in range(13):
            self.layersRDB.append(ResidualDenseBlock_5C(nf, gc))
        self.layersRDB = nn.Sequential(*self.layersRDB)

    def forward(self, x):
        out = self.layersRDB(x)
        return out * 0.2 + x

class RRDBBlock(nn.Module):
    def __init__(self, nf, nb, gc=32):
        super(RRDBBlock, self).__init__()
        self.layers = []
        for i in range(nb):
            self.layers.append(RRDB(nf, gc))

        self.layers = nn.Sequential(*self.layers)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
 
    def forward(self, x):
        out = self.lrelu((self.layers(x)))
        return out

class MetaRRDB(nn.Module):
    def __init__(self, args):
        super(MetaRRDB, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        conv=common.default_conv
        kernel_size = 3
        n_feats = 64
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        # number of RRDB blocks (nb), conv layers, out channels        
        self.nf = 64
        self.gc = 32
        self.nb = 24

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        
        # Redidual dense blocks and dense feature fusion
        self.RRDBgroup = RRDBBlock(self.nf, self.nb, self.gc)
        self.conv2 = nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1)

        self.upsampler = common.Upsampler(conv, self.args.scale2[0], n_feats, act=False) 
                                 
        self.tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*self.tail)

    def forward(self, x, pos_mat, a):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        
        x = self.RRDBgroup(x)
        x = self.conv2(x)

        out = self.upsampler(x)
        out = self.tail(out)

        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        #self.scale = self.args.scale[scale_idx]
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
        #print(self.scale)


