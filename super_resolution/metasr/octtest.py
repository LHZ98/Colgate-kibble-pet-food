import os

from data import common
from data import multiscalesrdatav2 as srdata
#from data import srdata as srdata
import numpy as np

import torch
import torch.utils.data as data

class OCTtest(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(OCTtest, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        
    def _scan(self):
        names_hr, names_lr = super(OCTtest, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data,'OCTtest', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)
