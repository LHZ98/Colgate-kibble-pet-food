""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
from .scale_attention_layer import scale_atten_convblock
from .modules import UnetDsv3

class UNet_atten(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_atten, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.out_size = (512, 512)
        # self.out_size = (512,int(2007*0.25))
        # self.out_size = (256, 256)

        self.inc = (DoubleConv(n_channels, 64))  # 64 channels
        self.down1 = (Down(64, 128))  # 128 channels
        self.down2 = (Down(128, 256))  # 256 channels
        self.down3 = (Down(256, 512))  # 512 channels
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))  # 1024 channels

        self.up4 = (Up(1024, 512 // factor, bilinear))  # 512 channels
        self.up3 = (Up(512, 256 // factor, bilinear))  # 256 channels
        self.up2 = (Up(256, 128 // factor, bilinear))  # 128 channels
        self.up1 = (Up(128, 64, bilinear))  # 64 channels
        self.outc = (OutConv(4, n_classes))
        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)

        # Deep supervision
        self.dsv4 = UnetDsv3(in_size=512, out_size=4, scale_factor=self.out_size)  # 4 channels
        self.dsv3 = UnetDsv3(in_size=256, out_size=4, scale_factor=self.out_size)  # 4 channels
        self.dsv2 = UnetDsv3(in_size=128, out_size=4, scale_factor=self.out_size)  # 4 channels
        self.dsv1 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)  # 4 channels
        
        

    def forward(self, x):
        x1 = self.inc(x)  # 64 channels

        x2 = self.down1(x1)  # 128 channels
        x3 = self.down2(x2)  # 256 channels
        x4 = self.down3(x3)  # 512 channels
        x5 = self.down4(x4)  # 1024 channels

        up4 = self.up4(x5, x4)  # 512 channels
        up3 = self.up3(up4, x3)  # 256 channels
        up2 = self.up2(up3, x2)  # 128 channels
        up1 = self.up1(up2, x1)  # 64 channels

        # Deep Supervision
        dsv4 = self.dsv4(up4)  # 4 channels
        dsv3 = self.dsv3(up3)  # 4 channels
        dsv2 = self.dsv2(up2)  # 4 channels
        dsv1 = self.dsv1(up1)  # 4 channels
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        logits = self.outc(out)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)