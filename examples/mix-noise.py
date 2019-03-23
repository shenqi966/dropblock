#coding:utf-8
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

from dropblock import DropBlock2D, LinearScheduler, DropBlock2DMix
from _lib.roi_align.crop_and_resize import CropAndResizeFunction, CropAndResize
import numpy as np
from torch.autograd import Variable
from RoIAlign.roi_align.roi_align import RoIAlign
import os
import torch.distributions as tdist

def test_noise():
    n = tdist.Normal(torch.tensor([0.5]), torch.tensor([0.5]))
    x = n.sample((1, 32, 32))
    print(x)


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size()) # 8 8

        # x, index = self.dropblockmix(self.layer1(x)) # print(x.size()) # 8 8
        x = self.layer1(x)
        x = self.layer2(x)        # print(x.size()) # 4 4
        x = self.layer3(x) # 2 2

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


def resnet9(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)



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
    parser.add_argument('--epochs', required=False, type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop_prob', required=False, type=float, default=0.,
                        help='dropblock dropout probability')
    parser.add_argument('--block_size', required=False, type=int, default=3,
                        help='dropblock block size')
    parser.add_argument('--device', required=False, default=0, type=int,
                        help='CUDA device id for GPU training')
    parser.add_argument('--tag', required=False, type=str, default='dbmix-default',
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

    # Add noise class
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'noise')

    # define model
    model = resnet9(num_classes=len(classes), drop_prob=drop_prob, block_size=block_size)
    model_name = model._get_name()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.cuda()
    best_acc = 0
    noise = tdist.Normal(torch.tensor([0.5]), torch.tensor([0.5]))
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        t0 = time.time()
        for batch_idx, data_batch in enumerate(train_loader):
            x, y = data_batch
            noise_x = noise.sample((x.size(0), 3, 32, 32)).squeeze(-1)
            # print(x.size(), type(x))
            # print(noise_x.size(), type(noise_x))
            yb = torch.LongTensor([10]*y.size(0))
            x = 0.99*x + 0.01*noise_x
            # print(y, yb)

            x = Variable(x).cuda()
            y = Variable(y).cuda()
            yb = Variable(yb).cuda()
            outputs = model(x)

            optimizer.zero_grad()

            loss1 = criterion(outputs, y)
            loss2 = criterion(outputs, yb)
            loss = 0.99*loss1 + 0.01*loss2

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(y.data).cpu().sum().item()
            total += bsize
            correct_percent = 1.0 * correct / total
            t1 = time.time()
            print("\r[EPOCH {0:d}] \tloss: {1:6.4f}, \tacc: {2:6.4f} \tdrop_prob:{3} \ttime:{4:2.4f}".format(epoch, train_loss / (batch_idx+1), correct_percent, drop_prob, (t1-t0)/(batch_idx+1)), end="\r")
        print("")

        model.eval()
        correct_test = 0
        total_test = 0
        for batch_idx, data_batch in enumerate(test_loader):
            x, y = data_batch
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)
            correct_test += predicted.eq(y.data).cpu().sum().item()
            total_test += bsize

        correct_percent = 1.0 * correct_test / total_test
        if correct_percent > best_acc:
            best_acc = correct_percent
        print("TEST acc: {0:6.4f},   BEST acc: {1:6.4f}".format(correct_percent, best_acc))
    filedir = options.tag
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filename = os.path.join(filedir, "{}_bs{}_dp{}_epoch{}.pth".format(model_name, bsize, drop_prob, epochs))
    print("Save to {}".format(filename))
    torch.save(model.state_dict(), filename)


    # 0.8143 for align (test with rand)
    # 0.8116 for align  (with somewrong)
    # 0.8142 for align

    # 0.8060 for baseline
    # 0.8090 for drop_prob 0.1
    # 0.8127 for drop_prob 0.25

    # dbmix is efficient 0-0.25 => 0.8205
    # Add exponent 0-0.25 => 0.8187
    # 8197 (* 0.1 ...
    # 82.41 for blocksize=3 ,*0.1,
    # 82.70 for blocksize=3, 没有0.1, 但是loss混合时加了
    # 82.26 for ... = 3, 调整loss大小

    # noise_mix 81.32 (around 80.8) for lam=0.01

    # Schedular is a efficient trick
