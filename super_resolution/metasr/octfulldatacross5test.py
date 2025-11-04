import os
from data import multiscalesrdatav2

class OCTFullDataCross5Test(multiscalesrdatav2.SRData):
    def __init__(self, args, name='OCTFullDataCross5Test', train=True, benchmark=False):
        super(OCTFullDataCross5Test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(OCTFullDataCross5Test, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(OCTFullDataCross5Test, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')

