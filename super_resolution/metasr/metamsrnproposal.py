from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_model(args, parent=False):
    return MetaMSRN(args)

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
        self.fusion = nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1)
        self.P2W = Pos2Weight(inC=n_feats, outC=args.n_colors)
        self.MRMparamteres = MRMparameters(inC=n_feats, outC=n_feats, kernelsize=args.n_MRMconvkernel, numbercovns=args.n_MRMconvs)
        
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
        import matplotlib.pyplot as plt
        import imageio
        C1Out = res.squeeze(0).cpu().detach()
        for j in range(C1Out.shape[0]):
            
            #normalized = C1Out[j].data.mul(255)
            normalized = C1Out[j].mul(255)
            #ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            ndarr = normalized.byte().permute(1, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format('Proposal/FeatureExtraction', j), ndarr)
            
            #fig = plt.figure()
            #plt.xticks(alpha=0)
            #plt.tick_params(axis='x', width=0)
            #plt.yticks(alpha=0)
            #plt.tick_params(axis='y', width=0)
            #plt.imshow(C1Out[j,:,:])
            #plt.savefig('Proposal/FeatureExtraction' + str(j) + '.jpg', bbox_inches='tight')
        #########################################
        weights = self.MRMparamteres(a)

        testtemp1w = nn.Parameter(weights.view(64,64,self.args.n_MRMconvkernel,self.args.n_MRMconvkernel), requires_grad=False)

        device = torch.device('cpu' if self.args.cpu else 'cuda')
        testtemp1b = nn.Parameter(torch.zeros(64).contiguous().to(device), requires_grad=False)

        x = F.conv2d(res, testtemp1w, testtemp1b, stride=1, padding=(self.args.n_MRMconvkernel - 1) // 2)
        #########################################
        C2Out = x.squeeze(0).cpu().detach()
        #print(C2Out.shape)
        for j in range(C2Out.shape[0]):
            normalized = C2Out[j].data.mul(255)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format('Proposal/SpectrumRestoration', j), ndarr)
            
            #fig = plt.figure()
            #plt.xticks(alpha=0)
            #plt.tick_params(axis='x', width=0)
            #plt.yticks(alpha=0)
            #plt.tick_params(axis='y', width=0)
            #plt.imshow(C2Out[j,:,:])
            #plt.savefig('Proposal/SpectrumRestoration' + str(j) + '.jpg', bbox_inches='tight')
        #########################################
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)
        
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,self.args.n_colors).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, self.args.n_colors)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,self.args.n_colors,x.size(2),x.size(3)).permute(0,3,4,1,5,2) #
        out = out.contiguous().view(x.size(0),self.args.n_colors, scale_int*x.size(2),scale_int*x.size(3)) #
        #print(out.shape)
        ########################################
        C3Out = out.squeeze(0).cpu().detach()
        #print(C3Out.shape)
        for j in range(C3Out.shape[0]):
            
            normalized = C3Out[j].data.mul(255)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format('Proposal/SpatialRestoration', j), ndarr)
            #fig = plt.figure()
            #plt.xticks(alpha=0)
            #plt.tick_params(axis='x', width=0)
            #plt.yticks(alpha=0)
            #plt.tick_params(axis='y', width=0)
            #plt.imshow(C3Out[j,:,:])
            #plt.savefig('Proposal/SpatialRestoration' + str(j) + '.jpg', bbox_inches='tight')
        return out

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale) 
        N,C,H,W = x.size() 
        x = x.view(N,C,H,1,W,1)
  
        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)
        return x.contiguous().view(-1, C, H, W)

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale2[self.scale_idx % len(self.args.scale2)]
