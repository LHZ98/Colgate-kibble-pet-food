from model import common
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def make_model(args, parent=False):
    return MetaRDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

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
        self.inC = inC
        self.outC = outC
        self.kernelsize = kernelsize
        self.numbercovns = numbercovns
        self.MRM_block=nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,  64*64*self.kernelsize*self.kernelsize)
        )
    def forward(self,x):
        output = self.MRM_block(x)
        return output

class MetaRDN(nn.Module):
    def __init__(self, args):
        super(MetaRDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (10, 6, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        ## position to weight
        self.P2W = Pos2Weight(inC=G0, outC=args.n_colors)
        ##
        self.MRMparamteres = MRMparameters(inC=G0, outC=G0, kernelsize=args.n_MRMconvkernel, numbercovns=args.n_MRMconvs)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat, a):
        #d1 =time.time()
        #x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        #########################################
        weights = self.MRMparamteres(a)

        testtemp1w = nn.Parameter(weights.view(64,64,self.args.n_MRMconvkernel,self.args.n_MRMconvkernel), requires_grad=False)

        device = torch.device('cpu' if self.args.cpu else 'cuda')
        testtemp1b = nn.Parameter(torch.zeros(64).contiguous().to(device), requires_grad=False)

        x = F.conv2d(x, testtemp1w, testtemp1b, stride=1, padding=(self.args.n_MRMconvkernel - 1) // 2)
        #########################################
        x += f__1

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        #print(d2)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)
        
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,self.args.n_colors).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, self.args.n_colors)
        print(cols.shape)
        print(local_weight.shape)
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


