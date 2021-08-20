import os
from data import srdata


class sun(srdata.SRData):
    def __init__(self, args, name='sun', train=True, benchmark=False):
        super(sun, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('.jpg', '.jpg')