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

class MetaRDU(nn.Module):
    def __init__(self, args):
        super(MetaRDU, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        
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
        self.P2W = Pos2Weight(inC=filters_0, outC=args.n_colors)
        ##
        self.MRMparamteres = MRMparameters(inC=filters_0, outC=filters_0, kernelsize=args.n_MRMconvkernel, numbercovns=args.n_MRMconvs)
        ##

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale) 
        N,C,H,W = x.size() 
        x = x.view(N,C,H,1,W,1)
  
        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

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
        weights = self.MRMparamteres(a)

        testtemp1w = nn.Parameter(weights.view(128,128,self.args.n_MRMconvkernel,self.args.n_MRMconvkernel), requires_grad=False)

        device = torch.device('cpu' if self.args.cpu else 'cuda')
        testtemp1b = nn.Parameter(torch.zeros(128).contiguous().to(device), requires_grad=False)

        out_7 = F.conv2d(out_6, testtemp1w, testtemp1b, stride=1, padding=(self.args.n_MRMconvkernel - 1) // 2)
        #########################################
        x = out_7 + inputs
        
        #print(x.shape)
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


