#coding:utf-8
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append("..")
print(sys.path)

from dropblock import DropBlock2D, LinearScheduler
from _lib.roi_align.crop_and_resize import CropAndResizeFunction, CropAndResize
import numpy as np
from torch.autograd import Variable
from RoIAlign.roi_align.roi_align import RoIAlign
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB_Mixed(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 1.0, visual = 1):
        super(BasicRFB_Mixed, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        print("in_planes, out_planes, inter_", in_planes, out_planes, inter_planes)
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch1_2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,dilation=visual + 1, relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1))

        )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )
        self.branch2_2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes // 2) * 3, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, relu=False)
                )

        self.ConvLinear = BasicConv(10*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1_2 = self.branch1_2(x)
        x2 = self.branch2(x)
        x2_2 = self.branch2_2(x)

        out = torch.cat((x0,x1,x1_2,x2,x2_2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_Rep(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 1.0, visual = 1):
        print("[DEBUG] Basic RFB Repetitive")
        super(BasicRFB_Rep, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        print("in_planes, out_planes, inter_", in_planes, out_planes, inter_planes)
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch1_2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )
        self.branch2_2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 1.0, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        print("in_planes, out_planes, inter_", in_planes, out_planes, inter_planes)
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


vggcfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_RFB':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 'Dconv', 'M', 512, 512, 512, 512, 'M'],   # 93.7
    'VGG19_MRFB': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 'Mconv', 'M', 512, 512, 512, 512, 'M'],  # 93.99
    'VGG19_RRFB': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 'Rconv', 'M', 512, 512, 512, 512, 'M'],  # 93.87
}



class VGG(nn.Module):
    def __init__(self, vgg_name, in_channel=3, out_channel=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(vggcfg[vgg_name], in_channel=in_channel)
        self.classifier = nn.Linear(512, out_channel)
        # self.extra_classifier = False
        # if out_channel != 10:
        #     self.extra_classifier = True
        #     self.classifier2 = nn.Linear(512, out_channel)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out1 = self.classifier(out)
        # if self.extra_classifier:
        #     out2 = self.classifier2(out)
        #     return out2
        return out1

    def _make_layers(self, cfg, in_channel=3):
        layers = []
        in_channels = in_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'Dconv':
                print("[DEBUG VGG MAKE LYAER]")
                layers += [BasicRFB(in_planes=in_channels, out_planes=512)]
                in_channels = 512
            elif x == 'Mconv':
                layers += [BasicRFB_Mixed(in_planes=in_channels, out_planes=512)]
                in_channels = 512
            elif x == 'Rconv':
                layers += [BasicRFB_Rep(in_planes=in_channels, out_planes=512)]
                in_channels = 512
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

class VGG2(nn.Module):
    def __init__(self, vgg_name='VGG19', in_channel=3, out_channel=10, drop_prob=0.0, block_size=5):
        super(VGG2, self).__init__()
        # 修改了 make_layer 这个部分， 源码是在上面， 把中间分层处理
        self.f1, self.f2, self.f3 = self._make_layers(vggcfg[vgg_name], in_channel=in_channel)
        self.classifier = nn.Linear(512, out_channel)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )

        self.wh = 2
        self.wh2 = 4
        self.align_sche = False
        self.i = 0

        # self.cr = CropAndResize(8, 8)  # 8 is according to the real size
        self.cr = RoIAlign(self.wh, self.wh, transform_fpcoor=True)
        self.cr2 = RoIAlign(self.wh2, self.wh2, transform_fpcoor=True)

        # 注释掉 其中的一个，
        # self.forward = self._forward_dropblock; print("-------  VGG with Dropblock  ---------\n")
        self.forward = self._forward_align; print("-------  VGG with ROiAlign  ---------\n")

    def _forward_dropblock(self, x):
        # out = self.features(x)
        # 这里添加的是 dropblock ， 用来对比
        out = self.f1(x)
        # print("1: ", out.size())  # 8 8
        out = self.dropblock(out)
        # print("2: ", out.size()) # 8 8
        out = self.f2(out)
        # print("3: ", out.size()) # 4 4
        out = self.dropblock(out)
        out = self.f3(out)
        # print("4: ", out.size()) # 1 1

        out = out.view(out.size(0), -1)
        out1 = self.classifier(out)
        return out1

    def _forward_align(self, x):
        # 添加 Roi Align
        init_param = 0.1
        if self.align_sche:
            vlist = np.linspace(1, 2, 5000)  # varies 1 ~ 2
            if self.i < len(vlist):
                param = vlist[self.i] * init_param
        else:
            param = init_param

        out = self.f1(x)

        out = self.f2(out) # 需要知道这里的size
        if self.training == True:
            # print(type(x))
            rs = np.random.random(4) * param * 0.5 ; #print(param)
            rs[2], rs[3] = 1 - rs[2], 1 - rs[3]
            rs *= self.wh2
            bs = x.size(0)
            bbox = to_varabile(np.asarray([rs], dtype=np.float32))
            bbox = bbox.repeat(bs, 1)
            # print(bbox)
            box_index_data = to_varabile(np.arange(bs, dtype=np.int32))
            x = self.cr2(x, bbox, box_index_data)
            # print("\r...                                   .... .....................................Runing Roialign", end="\r")
        else:
            pass
            # print("\r........                                     ....................................TEST No Align ", end="\r")
        out = self.f3(out)

        out = out.view(out.size(0), -1)
        out1 = self.classifier(out)
        return out1

    def _make_layers(self, cfg, in_channel=3):
        layers = []
        in_channels = in_channel
        for x in cfg[:6]:  # 3 6 11 16 19
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        re = nn.Sequential(*layers)

        layers2 = []
        for x in cfg[6:11]:  # 16 19
            if x == 'M':
                layers2 += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers2 += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers2 += [nn.AvgPool2d(kernel_size=1, stride=1)]
        re2 = nn.Sequential(*layers2)

        layers3 = []
        for x in cfg[11:]:  # 16 19
            if x == 'M':
                layers3 += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers3 += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers3 += [nn.AvgPool2d(kernel_size=1, stride=1)]
        re3 = nn.Sequential(*layers3)
        return re, re2, re3

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
