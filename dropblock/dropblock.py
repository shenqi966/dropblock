#coding:utf-8
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
import numpy as np

class DropBlock2D(nn.Module):
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

    def __init__(self, drop_prob, block_size, test=False, gkernel=False):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.test = test
        self.gkernel = gkernel
        if gkernel == True:
            print("[*] Using Gaussian-like Kernel")
            print("[*] Gkernel size =", block_size)
            x, y = np.meshgrid(np.linspace(-1, 1, block_size), np.linspace(-1, 1, block_size))
            d = np.sqrt(x * x + y * y)
            # hyper-parameter
            sigma, mu = 0.75, 0.0
            g = np.clip(np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * 1.25, 0.0, 1.0)
            self.g = g[np.newaxis, np.newaxis, :, :]


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

            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)
            if self.test: print("--- gamma ---\n", gamma)

            # sample mask
            mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))
            if self.test: print("---  mask ---\n", mask)

            # place mask on input device
            mask = mask.to(x.device)   # mask.cuda()

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            if self.test: print("--- block mask ---\n", block_mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):

        if self.gkernel == True:
            kernel = torch.from_numpy(self.g).to(mask.device)
        else:
            kernel = torch.ones((1, 1, self.block_size, self.block_size)).to(
                mask.device)
        block_mask = F.conv2d(mask[:, None, :, :],kernel,
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
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            mask_reduction = self.block_size // 2
            mask_depth = x.shape[-3] - mask_reduction
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_depth, mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)

            # sample mask
            mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv3d(mask[:, None, :, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size // 2) + 1))

        delta = self.block_size // 2
        input_depth = mask.shape[-3] + delta
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        depth_to_crop = block_mask.shape[-3] - input_depth
        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if depth_to_crop != 0:
            block_mask = block_mask[:, :, :-depth_to_crop, :, :]

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :, :-width_to_crop]

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_volume = x.shape[-3] * x.shape[-2] * x.shape[-1]
        mask_volume = mask_sizes[-3] * mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 3)) * (feat_volume / mask_volume)



class DropBlock2DMix(nn.Module):
    """
    DropBlock with mixing
    """
    def __init__(self, drop_prob, block_size, test=False, extra_mix=False):
        super(DropBlock2DMix, self).__init__()
        print("[*] using Dropblock mix")
        print("[***]  Setting fixed drop_window")
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.test = test
        self.extra_mix = extra_mix

    def forward(self, x, index=None):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            # raise ValueError("Dropblock mix, drop_prob > 0 ?")
            return x, None, None, None
        else:
            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            # gamma = self._compute_gamma(x, mask_sizes)
            # if self.test: print("--- gamma ---\n", gamma)
            # # sample mask
            # mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))
            # if self.test: print("---  mask ---\n", mask)
            bs = x.shape[0]
            hw = mask_width
            # rads = torch.randint(0, hw * hw, (bs,)).long()
            rads = torch.randint(0, hw * hw, (1,)).long().repeat(bs) # repeat mode
            rads = torch.unsqueeze(rads, 1)
            mask = torch.zeros(bs, hw*hw).scatter_(1, rads, 1).reshape((bs,hw,hw))

            # place mask on input device
            mask = mask.to(x.device)   # mask.cuda()

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            if self.test: print("--- block mask ---\n", block_mask)

            # apply block mask
            # out = x * block_mask[:, None, :, :]

            batch_size = x.size()[0]
            if index == None:
                index = torch.randperm(batch_size).cuda()
            verse_mask = torch.ones_like(block_mask) - block_mask
            if self.test: print("--- verse_mask ---", verse_mask)

            if self.extra_mix:
                lam = 0.05
                out = x*block_mask[:, None, :, :]*(1-lam) + \
                      x*verse_mask[:, None, :, :]*lam + \
                      x[index, :]*block_mask[:, None, :, :]*(lam) + \
                      x[index, :]*verse_mask[:, None, :, :]*(1-lam)
            else:
                out = x * block_mask[:, None, :, :] + \
                      x[index, :] * verse_mask[:, None, :, :] #* 0.1 这里需注意，是否加0.1
            # if self.test: out = x * block_mask[:, None, :, :] + x[index, :] * verse_mask[:, None, :, :] * 0.1
            # scale output
            # out = out * block_mask.numel() / block_mask.sum()

            return out, index, block_mask, verse_mask

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
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)

class DropChannel(nn.Module):
    """
    DropBlock with mixing
    """
    def __init__(self, drop_prob, test=False, extra_mix=False):
        super(DropChannel, self).__init__()
        print("[*] using Drop Channel")
        self.drop_prob = drop_prob
        # self.block_size = block_size
        self.test = test
        self.extra_mix = extra_mix

    def forward(self, x, index=None):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            # raise ValueError("Dropblock mix, drop_prob > 0 ?")
            # print("On Testing")
            return x
        else:
            bs = x.shape[0]
            c = x.shape[1]
            h, w = x.shape[-1], x.shape[-2]
            index = torch.unsqueeze(Bernoulli(1.0 - self.drop_prob).sample((bs, c,)) , 2)
            mask = index.repeat(1,1,h*w).reshape(bs,c,h,w).to(x.device)

            out = x * mask

            return out


class DropCBlock(nn.Module):
    def __init__(self, drop_prob, block_size, test=False):
        super(DropCBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.test = test
        print("[*] Using Drop Cblock ``")

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
            # get gamma value
            gamma = self._compute_gamma(x, mask_sizes)
            # sample mask
            mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))

            # place mask on input device
            mask = mask.to(x.device)   # mask.cuda()

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            channel_mask = self._compute_channel_mask(x, block_mask)
            # apply block mask
            out = x * channel_mask

            return out

    def _compute_block_mask(self, mask):
        kernel = torch.ones((1, 1, self.block_size, self.block_size)).to(
                mask.device)
        block_mask = F.conv2d(mask[:, None, :, :],kernel,
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
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)

    def _compute_channel_mask(self, x, block_mask):

        bs = x.shape[0]
        c = x.shape[1]
        h, w = x.shape[-1], x.shape[-2]
        index = torch.unsqueeze(Bernoulli(1.0 - self.drop_prob).sample((bs, c,)), 2)
        mask = index.repeat(1, 1, h * w).reshape(bs, c, h, w).to(x.device)
        mask = mask * (1 - block_mask)[:, None, :, :]
        mask = 1 - mask
        # print("c mask", mask)
        # print("block mask", block_mask)
        return mask


if __name__ == "__main__":
    db = DropBlock2DMix(0.2, 3, True)
    cb = DropCBlock(0.2, 3)
    from torch.autograd import Variable
    import numpy as np
    hw = 6
    x = torch.Tensor(np.arange(hw*hw*4).reshape((1,4,hw,hw)))
    x = Variable(x)
    # xx, index = db(x)
    xx = cb(x)
    # print(xx, xx.size())
    # print(index)