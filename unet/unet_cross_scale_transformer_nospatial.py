""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self,num_pos_feats=512,len_embedding=32):
        super(PositionalEncoding,self).__init__()
        self.row_embed = nn.Embedding(len_embedding,num_pos_feats).requires_grad_(True)
        # self.col_embed = nn.Embedding(len_embedding,num_pos_feats).requires_grad(False)
        self.num_pos_feats = num_pos_feats
        self.reset_parameters()
        # self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
    
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight) #uniform to 0 to 1
        # nn.init.uniform_(self.col_embed.weight)
    
    def create_circular_matrix(self,height, width):
        center_i, center_j = height // 2, width // 2
        matrix = np.zeros((height, width), dtype=int)

        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                matrix[i, j] = int(distance)

        return matrix

    def create_shortest_path_matrix(self, height,width):
        center_x, center_y = width // 2, height // 2
        matrix = torch.zeros((height, width), dtype=int)

        for i in range(height):
            for j in range(width):
                matrix[i, j] = max(abs(center_x - j) + abs(center_y - i) - 1, 0)

        return matrix

    def forward(self,x):

        # #circular position encoding
        h,w = x.shape[-2:]
        Z = self.create_shortest_path_matrix(h,w).to(x.device)
        # print('Z shape:',np.shape(Z))

        pos = torch.empty(size=(h,w,self.num_pos_feats),device = x.device)
        # print(pos.size())

        for i in range(h):
            # print('i:',i)
            # print('Zi:',(Z[i,:].size()))

            # t = torch.tensor(Z[i,:],device = x.device)
        #     print('tensor t:',t)
            # embed = self.row_embed(Z[i,:])
            # print('embed:',embed.size())
            # embed = embed[:,:,None]
            # embed = self.conv1(embed)
            pos[i,:,:] = self.row_embed(Z[i,:])

        pos = pos.permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)
        # pos = self.conv1(pos)
        # print(pos.size())

        #two dimension position encoding
        # h,w = x.shape[-2:]
        # i = torch.arange(w,device = x.device)
        # j = torch.arange(h,device = x.device) #generate [0,h-1] position code


        # x_embed = self.col_embed(i)
        # y_embed = self.row_embed(j)

        # # print('x_embed:',x_embed.size())
        # # print('y_embed:',y_embed.size())

        # pos = torch.cat([
        #     x_embed.unsqueeze(0).repeat(h,1,1),
        #     y_embed.unsqueeze(1).repeat(1,w,1),
        #     ],dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)

        # print('pos size:',pos.size())

        return pos

class spatial_atten_module(nn.Module):
    def __init__(self, in_dim):
        super(spatial_atten_module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class self_atten_module(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output


class Unet_cross_scale_transformer_nospatial(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_cross_scale_transformer_nospatial, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        # # positional encoding
        # self.pe1 = PositionalEncoding(64)

        # #spatial attention
        # self.spatial_atten1 = spatial_atten_module(64)

        # #self attention
        # self.self_atten1 = self_atten_module(64)



        # positional encoding
        # self.pe3 = PositionalEncoding(256)

        #spatial attention
        # self.spatial_atten3 = spatial_atten_module(256)

        #self attention
        # self.self_atten3 = self_atten_module(256)

        # self.spatial_atten4 = spatial_atten_module(512)


        # positional encoding
        self.pe = PositionalEncoding(1024)

        #spatial attention
        # self.spatial_atten = spatial_atten_module(1024)

        #self attention
        self.self_atten = self_atten_module(1024)



    def forward(self, x):
        x1 = self.inc(x)
        # x1_spatial_atten = self.spatial_atten1(x1)
        # x1 = x1_spatial_atten

        # x1_pe = self.pe1(x1)
        # x1=x1+x1_pe
        # x1_self_atten = self.self_atten1(x3)
        # x1 = x1_spatial_atten+x1

        x2 = self.down1(x1)

        x3 = self.down2(x2)
        # print('x3:',x3.size())

        # x3_spatial_atten = self.spatial_atten3(x3)
        # x3_pe = self.pe3(x3)
        # x3=x3+x3_pe
        # x3_self_atten = self.self_atten3(x3)
        # x3 = x3_spatial_atten+x3
        # x3 = x3_spatial_atten+x3

        # print('x3 now:',x3.size())
        x4 = self.down3(x3)
        # x4_spatial_atten = self.spatial_atten4(x4)
        # x4 = x4_spatial_atten+x4

        x5 = self.down4(x4)
        
        # print('x1:',x1.size())
        # print('x2:',x2.size())
        # print('x3:',x3.size())
        # print('x4:',x4.size())
        # print('x5:',x5.size())

        # x5_spatial_atten = self.spatial_atten(x5)
        # print('x5_spatial_atten:',x5_spatial_atten.size())

        x5_pe = self.pe(x5)
        x5=x5+x5_pe
        # print(x5_pe)
        # print('x5:',x5.size())

        x5 = self.self_atten(x5)
        # print('x5_self_atten:',x5_self_atten.size())

        # print('x5:',x5.size())

        x = self.up1(x5, x4)
        # print('xup1:',x.size())
       
        x = self.up2(x, x3)
        # print('xup2:',x.size())

        x = self.up3(x, x2)
        # print('xup3:',x.size())

        x = self.up4(x, x1)
        # print('xup4:',x.size())

        logits = self.outc(x)
        # print('logits:',logits.size())
        return logits

        # x6 = self.up1(x5, x4)
        # x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        # x6_cat = torch.cat((x5_scale, x6), 1)

        # x7 = self.up2(x6_cat, x3)
        # x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        # x7_cat = torch.cat((x6_scale, x7), 1)

        # x8 = self.up3(x7_cat, x2)
        # x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        # x8_cat = torch.cat((x7_scale, x8), 1)

        # x9 = self.up4(x8_cat, x1)
        # x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        # x9 = torch.cat((x8_scale, x9), 1)

        # logits = self.outc(x9)

        
    #     # return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)