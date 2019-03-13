import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        out, index = self.dropblock(x)
        return out, index

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

if __name__ == "__main__":
    # from .dropblock import DropBlock2D
    # ls = LinearScheduler(
    #         DropBlock2D(drop_prob=0.25, block_size=5),
    #         start_value=0.,
    #         stop_value=0.25,
    #         nr_steps=5e3
    #     )
    dv = np.linspace(start=0., stop=0.25, num=5e3)
    print(dv)