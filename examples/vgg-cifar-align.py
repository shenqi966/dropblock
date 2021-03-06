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
from _lib.roi_align.crop_and_resize import CropAndResizeFunction, CropAndResize
import numpy as np
from torch.autograd import Variable
from RoIAlign.roi_align.roi_align import RoIAlign
import os
from vgg import VGG2

from tensorboardX import SummaryWriter

results = []

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

        self.wh = 4
        self.align_sche = True
        self.i = 0
        # self.cr = CropAndResize(8, 8)  # 8 is according to the real size

        self.cr = RoIAlign(self.wh, self.wh, transform_fpcoor=True)
        print("--------------------------------------------------------"
              "\n--------     RoiAlign                          -------\n"
              "--------------------------------------------------------")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # add crop_and_resize Here

        x = self.dropblock(self.layer1(x)); # print(x.size()) 8 8
        x = self.dropblock(self.layer2(x)); # print(x.size()) 4 4

        init_param = 0.05
        if self.align_sche:
            vlist = np.linspace(1,3,5000)
            if self.i < len(vlist):
                param = vlist[self.i] * init_param
        else: param = init_param

        if self.training == True:
            # print(type(x))
            rs = np.random.random(4) * param
            rs[2], rs[3] = 1 - rs[2], 1 - rs[3]
            rs *= self.wh
            bs = x.size(0)
            bbox = to_varabile(np.asarray([rs], dtype=np.float32))
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

        x = self.layer3(x);  # print(x.size())  # 2 2
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet9Align(**kwargs):
    return ResNetAlign(BasicBlock, [1, 1, 1, 1], **kwargs)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = 0.1
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (0.2 - 0.1) * epoch / 10.
    if epoch >= 80:
        lr /= 10
    if epoch >= 130:
        lr /= 10
    print("Learning rate = ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=128,   #
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=250,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--drop_prob', required=False, type=float, default=0.,
                        help='dropblock dropout probability')
    parser.add_argument('--block_size', required=False, type=int, default=5,
                        help='dropblock block size')
    parser.add_argument('--device', required=False, default=0, type=int,
                        help='CUDA device id for GPU training')
    parser.add_argument('--tag', required=False, type=str, default='vgg-align-16_2',
                        help='saving floder')
    parser.add_argument('--verbose', required=False, type=bool, default=False,
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
    verbose = options.verbose
    device = 'cpu' if options.device is None \
        else torch.device('cuda:{}'.format(options.device))

    # 标准化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define model
    model = VGG2(out_channel=len(classes), drop_prob=drop_prob, block_size=block_size, forward='align')
    model_name = model._get_name()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) # default SGD optim

    filedir = options.tag
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    writer = SummaryWriter(
        log_dir=os.path.join(filedir, "{}_bs{}_dp{}_epoch{}".format(model_name, bsize, drop_prob, epochs)))

    model.cuda()
    best_acc = 0
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        t0 = time.time()
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, data_batch in enumerate(train_loader):
            x, y = data_batch
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            outputs = model(x)
            optimizer.zero_grad()

            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(y.data).cpu().sum().item()
            total += bsize
            correct_percent = 1.0 * correct / total
            t1 = time.time()
            if verbose: print("\r[EPOCH {0:d}] \tloss: {1:6.4f}, \tacc: {2:6.4f} \tdrop_prob:{3} \ttime:{4:2.4f}".format(epoch, train_loss / (batch_idx+1), correct_percent, drop_prob, (t1-t0)/(batch_idx+1)), end="\r")
        print("")
        writer.add_scalar("train/loss", train_loss / (batch_idx+1), epoch)
        writer.add_scalar("train/acc", correct_percent, epoch)

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
        if verbose: print("TEST acc: {0:6.4f},   BEST acc: {1:6.4f}".format(correct_percent, best_acc))
        writer.add_scalar("test/acc", correct_percent, epoch)
        writer.add_scalar("test/best_acc", best_acc, epoch)

    filename = os.path.join(filedir, "{}_bs{}_dp{}_epoch{}.pth".format(model_name, bsize, drop_prob, epochs))
    print("Save to {}".format(filename))

    torch.save(model.state_dict(), filename)


    # 0.8143 for align (test with rand)
    # 0.8116 for align  (with somewrong)
    # 0.8142 for align

    # 0.8060 for baseline
    # 0.8090 for drop_prob 0.1
    # 0.8127 for drop_prob 0.25

    # Schedular is a efficient trick
