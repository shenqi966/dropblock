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

from dropblock import DropBlock2D, LinearScheduler, DropChannel, DropCBlock
from _lib.roi_align.crop_and_resize import CropAndResizeFunction, CropAndResize
import numpy as np
from torch.autograd import Variable
from RoIAlign.roi_align.roi_align import RoIAlign
import os

results = []


class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5, gkernel=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size, gkernel=gkernel),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.dropcblock = LinearScheduler(
            DropCBlock(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        # self.dropchannel = DropChannel(drop_prob=drop_prob)
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
        self.dropcblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size()) # 8 8

        x = self.dropcblock(self.layer1(x))
        # x = self.layer1(x)
        # print(x.size()) # 8 8
        x = self.dropcblock(self.layer2(x))
        # print(x.size()) # 4 4
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
    parser.add_argument('--epochs', required=False, type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop_prob', required=False, type=float, default=0.,
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

    model.cuda()
    best_acc = 0
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        t0 = time.time()
        alpha = 1.0
        for batch_idx, data_batch in enumerate(train_loader):
            x, y = data_batch
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(x.size(0)).cuda()
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]

            outputs = model(x)
            optimizer.zero_grad()

            # loss = criterion(outputs, y)
            loss_1 = criterion(outputs, y_a)
            loss_2 = criterion(outputs, y_b)
            loss = loss_1 * lam + loss_2 * (1-lam)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(y.data).cpu().sum().item()
            total += bsize
            correct_percent = 1.0 * correct / total
            t1 = time.time()
            print("\r[MIXING] [EPOCH {0:d}] \tloss: {1:6.4f}, \tacc: {2:6.4f} \tdrop_prob:{3} \ttime:{4:2.4f}".format(epoch, train_loss / (batch_idx+1), correct_percent, drop_prob, (t1-t0)/(batch_idx+1)), end="\r")
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

    # gkernel-5 0.8061 (around 0.802)  with default schedule with 2 dropgkernel
    # gkenrel-7 0.8103 (around 0.803)  with default schedule with 2 dropgkernel

    # dropchannel 0.8016 (around 0.785) with no schedule and 1 dropchannel (train-acc 0.91)

    # Schedular is a efficient trick
