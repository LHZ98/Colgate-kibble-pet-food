from model import common
import torch
import torch.nn as nn

# The Discriminator
class Discriminator(nn.Module):
    def __init__(self, args, gan_type):
      """
      Builds the Discriminator network.
      :param in_channels: Number of input channels. If RGB images, use 3, if grayscale, use 1.
      """
      super(Discriminator,self).__init__()
      self.in_channels = args.n_colors
      self.build_discriminator()
    
    def build_discriminator(self):
      """
      Actually builds the Discriminator network. Includes 4 blocks. The first two blocks
      are convolutions with LeakyReLU. Block 3 is a bunch of convolutions. Block 4
      is a pooling layer.
      """
      block1 = [nn.Conv2d(self.in_channels,64,kernel_size=3,padding=1),
              nn.LeakyReLU(0.2)]
      
      #block2 = [nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
      #        nn.BatchNorm2d(64),
      #        nn.LeakyReLU(0.2)]
      #block2 = [nn.utils.weight_norm(nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)),
      #        nn.InstanceNorm2d(64),
      #        nn.LeakyReLU(0.2)]
      block2 = [nn.utils.weight_norm(nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)),
              nn.LeakyReLU(0.2)]

      block3 = []
      for i in range(3):
        prev = 64 * (1 << i)
        next = 64 * (1 << (i+1))
        block3.append(nn.Conv2d(prev, next, kernel_size=3,padding=1))
        #block3.append(nn.InstanceNorm2d(next))
        #block3.append(nn.BatchNorm2d(next))
        block3.append(nn.LeakyReLU(0.2))

        block3.append(nn.Conv2d(next,next, kernel_size=3,stride=2,padding=1))
        #block3.append(nn.InstanceNorm2d(next))
        #block3.append(nn.BatchNorm2d(next))
        block3.append(nn.LeakyReLU(0.2))

      block4 = [nn.AdaptiveAvgPool2d(1),
              nn.Conv2d(512, 1024, kernel_size=1),
              nn.LeakyReLU(0.2),
              nn.Conv2d(1024, 1, kernel_size=1)]    

      self.block1 = nn.Sequential(*block1)
      self.block2 = nn.Sequential(*block2)
      self.block3 = nn.Sequential(*block3)
      self.block4 = nn.Sequential(*block4)
      self.lastAct = nn.LeakyReLU(0.2)

    def forward(self,x):
      """
      Forward propagates through the network.
      :param x: input image.
      :return: binary classification of image, either fake (0) or real (1).
      """
      batch_size = x.size(0)
      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)
      x = self.block4(x)
      x = self.lastAct(x)
      #return torch.sigmoid(x.reshape(batch_size))
      return x.reshape((batch_size, -1))