import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet
# from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
# from ignite.metrics import Accuracy
# from ignite.metrics import RunningAverage
# from ignite.contrib.handlers import ProgressBar

import sys
sys.path.append("..")
print(sys.path)

from dropblock import DropBlock2D, LinearScheduler
from dropblock.dropheat import DropHeat2D
from _lib.roi_align.crop_and_resize import CropAndResizeFunction, CropAndResize
import numpy as np
from torch.autograd import Variable
from RoIAlign.roi_align.roi_align import RoIAlign
import os

from tensorboardX import SummaryWriter

results = []

# class AddConv(nn.Module):
#     def __init__(self, in_planes, out_planes=1, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(AddConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         # self.conv = nn.Conv2d()
#     def forward(self, x):
#         x = self.conv(x)
#         return x

class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.dropheat = DropHeat2D(drop_prob=drop_prob, block_size=block_size)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.addconv = nn.Conv2d(64, 1, kernel_size=3, bias=False, padding=1)
        self.addconv.weight.data.fill_(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, writer=None):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size()) # 8 8

        x = self.dropheat(self.layer1(x))

        # print(x.size()) # 8 8
        x = self.dropblock(self.layer2(x))
        # print(x.size()) # 4 4
        x = self.layer3(x) # 2 2
        # print(x.size())

        x = self.layer4(x) # 1 1
        # print(x.size())


        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class ResNetAlign(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.wh = 2
        self.wh2 = 8
        self.align_sche = False
        self.i = 0
        # self.cr = CropAndResize(8, 8)  # 8 is according to the real size

        self.cr  = RoIAlign(self.wh, self.wh, transform_fpcoor=True)
        self.cr2 = RoIAlign(self.wh2, self.wh2, transform_fpcoor=True)
        print("--------------------------------------------------------"
              "\n--------     RoiAlign                          -------\n"
              "--------------------------------------------------------")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, writer=None):
        # self.dropblock.step()  # increment number of iterations

        init_param = 0.1
        if self.align_sche:
            vlist = np.linspace(1,2,5000)
            if self.i < len(vlist):
                param = vlist[self.i] * init_param
        else: param = init_param

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # add crop_and_resize Here
        # print(x.size())
        writer.add_image("features", x[0,0,:,:], 3)

        x = self.dropblock(self.layer1(x)); #print(x.size()) # 8 8
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

        x = self.dropblock(self.layer2(x)); # print(x.size()) 4 4
        x = self.layer3(x);  # print(x.size())  # 2 2

        if self.training == True:
            # print(type(x))
            rs = np.random.random(4) * param
            rs[2], rs[3] = 1 - rs[2], 1 - rs[3]
            rs *= 2
            bs = x.size(0)
            bbox = to_varabile(np.asarray([rs], dtype=np.float32)); # print(bbox)
            bbox = bbox.repeat(bs, 1)
            # print(bbox)
            box_index_data = to_varabile(np.arange(bs, dtype=np.int32))
            x = self.cr(x, bbox, box_index_data)
            # print("\r...                                   .... .....................................Runing Roialign", end="\r")
        else:
            # ss = 0
            # rs = np.array([ss,ss,ss,ss]) * 0.05
            # rs[2], rs[3] = 1 - rs[2], 1 - rs[3]
            # rs *= self.wh
            # bs = x.size(0)
            # bbox = to_varabile(np.asarray([rs], dtype=np.float32))
            # bbox = bbox.repeat(bs, 1)
            # # print(bbox)
            # box_index_data = to_varabile(np.arange(bs, dtype=np.int32))
            # x = self.cr(x, bbox, box_index_data)
            pass
            # print("\r........                                     ....................................TEST No Align ", end="\r")

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

def resnet9(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)

def resnet9Align(**kwargs):
    return ResNetAlign(BasicBlock, [1, 1, 1, 1], **kwargs)


def logger(engine, model, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    pbar.log_message(
        "Test Results - Avg accuracy: {:.2f}, drop_prob: {:.2f}".format(avg_accuracy,
                                                                        model.dropblock.dropblock.drop_prob)
    )
    results.append(avg_accuracy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=256,
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=150,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop_prob', required=False, type=float, default=0.8,
                        help='dropblock dropout probability')
    parser.add_argument('--block_size', required=False, type=int, default=5,
                        help='dropblock block size')
    parser.add_argument('--device', required=False, default=0, type=int,
                        help='CUDA device id for GPU training')
    parser.add_argument('--tag', required=False, type=str, default='default',
                        help='saving floder')
    options = parser.parse_args()
    print("Options: ")
    print(options)

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    lr = options.lr
    drop_prob = options.drop_prob
    block_size = options.block_size
    device = 'cpu' if options.device is None \
        else torch.device('cuda:{}'.format(options.device))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define model
    model = resnet9(num_classes=len(classes), drop_prob=drop_prob, block_size=block_size)
    model_name = model._get_name()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    filedir = options.tag
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    writer = SummaryWriter(log_dir=os.path.join(filedir, "{}_bs{}_dp{}_epoch{}".format(model_name, bsize, drop_prob, epochs)))

    filename = os.path.join(filedir, "{}_bs{}_dp{}_epoch{}.pth".format(model_name, bsize, drop_prob, epochs))
    print("Save Dir {}".format(filename))

    model.cuda()
    # model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


    best_acc = 0

    # model.eval()
    model.train()
    correct_test = 0
    total_test = 0
    for batch_idx, data_batch in enumerate(test_loader):
        x, y = data_batch
        img = x[0]
        # print(y[0])
        writer.add_image('img', img, 3)
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        outputs = model(x, writer)

        _, predicted = torch.max(outputs.data, 1)
        correct_test += predicted.eq(y.data).cpu().sum().item()
        total_test += bsize
        print("Exit(0)")
        exit(0)

    correct_percent = 1.0 * correct_test / total_test
    if correct_percent > best_acc:
        best_acc = correct_percent
    print("TEST acc: {0:6.4f},   BEST acc: {1:6.4f}".format(correct_percent, best_acc))




    # 0.8143 for align (test with rand)
    # 0.8116 for align  (with somewrong)
    # 0.8142 for align

    # 0.8060 for baseline
    # 0.8090 for drop_prob 0.1
    # 0.8127 for drop_prob 0.25

    # Schedular is a efficient trick
