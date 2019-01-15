import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropHeat2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size, test=False, power=1.0):
        super(DropHeat2D, self).__init__()

        self.drop_prob = drop_prob
        assert block_size == 5, "Not complete for block_size={}".format(block_size)
        # considering the blocksize of 5 firstly

        self.block_size = block_size
        self.test = test
        self.power = power
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.save_bm_init = 0.0
        self.save_bm = None
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            ##  custom droping
            w = self.avgpool(x)
            w.detach()
            w = w.sum(1)

            if self.test: print("X: ", x.size())
            # print("Y: ", w, w.size())

            w = w.view(w.shape[0], -1)
            minn = w.min(1)[0]  # minimum

            rd = torch.rand_like(w)  # return the rands  sized as w
            w = w - minn[:, None]
            if self.power != 1.0:
                w = w ** self.power
            meann = w.mean()
            w = w / meann

            # get gamma value

            gamma = self._compute_gamma(x, mask_sizes)
            if self.test: print("--- gamma ---\n", gamma)

            w = w * gamma
            mask = torch.le(rd, w).float()
            mask = mask.view(x.shape[0], mask_height, mask_width)
            if self.test: print("mask: ", mask)

            # # sample mask
            # mask_example = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))
            # assert mask.size() == mask_example.size(), "Size? {} and {}".format(mask.size(), mask_example.size())
            # assert type(mask) == type(mask_example), "Type? {} and {}".format(type(mask), type(mask_example))

            # place mask on input device
            mask = mask.to(x.device)   # mask.cuda()
            if True: # # debug
                if self.save_bm_init < 1.0 :
                    self.save_bm = mask.sum(0).data
                    self.save_bm_init += 1
                else:
                    self.save_bm *= self.save_bm_init
                    self.save_bm += mask.sum(0).data
                    self.save_bm_init += 1
                    self.save_bm /= self.save_bm_init
                # print(self.save_bm)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            if self.test: print("--- block mask ---\n", block_mask, block_mask.size())


            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            # print("dropheat ??? \r")
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv2d(mask[:, None, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size // 2) + 1))

        delta = self.block_size // 2
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_area = x.shape[-2] * x.shape[-1]
        mask_area = mask_sizes[-2] * mask_sizes[-1]
        if self.test: print("compute_gamma: ", self.drop_prob ,self.block_size,feat_area, mask_area)
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)



if __name__ == "__main__":
    db = DropHeat2D(0.4, 5, True)
    from torch.autograd import Variable
    import numpy as np
    x = torch.Tensor(np.arange(16*16).reshape((1,1,16,16)))
    x = Variable(x)
    xx = db(x)
    # print(xx)