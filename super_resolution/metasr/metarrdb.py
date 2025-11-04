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

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(4,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
        
    def forward(self,x):
        output = self.meta_block(x)
        return output

class MRMparameters(nn.Module):
    def __init__(self, inC, outC, kernelsize=3, numbercovns=3):
        super(MRMparameters,self).__init__()
        self.kernelsize = kernelsize
        self.numbercovns = numbercovns
        self.MRM_block=nn.Sequential(
            nn.Linear(2,inC),
            nn.ReLU(inplace=True),
            nn.Linear(inC,  inC*outC*self.kernelsize*self.kernelsize)
        )
    def forward(self,x):
        output = self.MRM_block(x)
        return output

class MetaRRDB(nn.Module):
    def __init__(self, args):
        super(MetaRRDB, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
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
        ## position to weight
        self.P2W = Pos2Weight(inC=self.nf, outC=args.n_colors)
        ##
        self.MRMparamteres = MRMparameters(inC=self.nf, outC=self.nf, kernelsize=args.n_MRMconvkernel, numbercovns=args.n_MRMconvs)
        ##

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale) 
        N,C,H,W = x.size() 
        x = x.view(N,C,H,1,W,1)
  
        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat, a):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        
        x = self.RRDBgroup(x)
        x = self.conv2(x)
        #########################################
        weights = self.MRMparamteres(a)

        testtemp1w = nn.Parameter(weights.view(64,64,self.args.n_MRMconvkernel,self.args.n_MRMconvkernel), requires_grad=False)

        device = torch.device('cpu' if self.args.cpu else 'cuda')
        testtemp1b = nn.Parameter(torch.zeros(64).contiguous().to(device), requires_grad=False)

        out_7 = F.conv2d(x, testtemp1w, testtemp1b, stride=1, padding=(self.args.n_MRMconvkernel - 1) // 2)
        #########################################
        x = out_7 + f__1
        
        #print(x.shape)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        #print(d2)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)
        
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,self.args.n_colors).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, self.args.n_colors)
        #print(cols.shape)
        #print(local_weight.shape)
        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,self.args.n_colors,x.size(2),x.size(3)).permute(0,3,4,1,5,2) #
        out = out.contiguous().view(x.size(0),self.args.n_colors, scale_int*x.size(2),scale_int*x.size(3)) #
        #out = self.add_mean(out) 

        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        #self.scale = self.args.scale[scale_idx]
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
        #print(self.scale)


