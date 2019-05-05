import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import math


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
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

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet18 = ResNet(Bottleneck, [3,4,6,3])

use_gpu = torch.cuda.is_available()
if use_gpu:
	resnet18 = resnet18.cuda()
	print ('USE GPU')
else:
	print ('USE CPU')

class MyDataset(torch.utils.data.TensorDataset):
    """Face Landmarks dataset."""

    def __init__(self,*tensors, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.img = tensors[0]
        self.target = tensors[1]


    def __getitem__(self, index):
        if not self.transform:
            return tuple(tensor[index] for tensor in self.tensors)
        return (self.transform(self.img[index]), self.target[index])

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(resnet18.parameters(), lr = 0.3, momentum = 0.1, weight_decay = 0.0001)
optimizer2 = optim.SGD(resnet18.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
optimizer3 = optim.SGD(resnet18.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0001)
optimizer4 = optim.SGD(resnet18.parameters(), lr = 0.0001, momentum = 0.9, weight_decay = 0.0001)
print ("1. Loading data")
import pandas as pd
images = pd.read_pickle('./train_images.pkl')
labels = pd.read_csv('./train_labels.csv', usecols = ['Category'])

print ("2. Converting data")
train_images = torch.from_numpy(np.array(images[:39000])).type(torch.LongTensor).view(-1, 1,64,64).float()
test_images = torch.from_numpy(np.array(images[39000:])).type(torch.LongTensor).view(-1, 1,64,64).float()
train_labels = torch.from_numpy(np.array(labels[:39000])).type(torch.LongTensor).view(-1)
test_labels = torch.from_numpy(np.array(labels[39000:])).type(torch.LongTensor).view(-1)
print (train_images.size(), train_labels.size())
resnet18.train()

import torchvision
MyTransform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    #torchvision.transforms.RandomRotation(20),
    # torchvision.transforms.RandomAffine(degrees = 0, shear=15),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
])

train = MyDataset(train_images,train_labels, transform = MyTransform)
#test = torch.utils.data.TensorDataset(test_images, test_labels)
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = False)
#test_loader = torch.utils.data.DataLoader(test, batch_size = 128, shuffle = False)

TestTransform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
])
test = MyDataset(test_images, test_labels, transform = TestTransform)
test_loader = torch.utils.data.DataLoader(test, batch_size = 128, shuffle = False)


print ("3. Training phase")
nb_train = train_images.shape[0]
nb_epoch = 20
nb_index = 0
nb_batch = 250

for epoch in range(15):
    resnet18.train()
    for t, (data, target) in enumerate(train_loader):
        optimizer1.zero_grad()
        data,target = Variable(data),Variable(target)
        data = data.cuda()
        target = target.cuda()
        pred = resnet18(data)
        loss = criterion(pred,target)
        loss.backward()
        optimizer1.step()
    resnet18.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = resnet18(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(epoch,t,loss.data, correct)

for epoch in range(15):
    resnet18.train()
    for t, (data, target) in enumerate(train_loader):
        optimizer2.zero_grad()
        data,target = Variable(data),Variable(target)
        data = data.cuda()
        target = target.cuda()
        pred = resnet18(data)
        loss = criterion(pred,target)
        loss.backward()
        optimizer2.step()
    resnet18.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = resnet18(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(epoch,t,loss.data, correct)

for epoch in range(15):
    resnet18.train()
    for t, (data, target) in enumerate(train_loader):
        optimizer3.zero_grad()
        data,target = Variable(data),Variable(target)
        data = data.cuda()
        target = target.cuda()
        pred = resnet18(data)
        loss = criterion(pred,target)
        loss.backward()
        optimizer3.step()
    resnet18.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = resnet18(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(epoch,t,loss.data, correct)
    
for epoch in range(15):
    resnet18.train()
    for t, (data, target) in enumerate(train_loader):
        optimizer4.zero_grad()
        data,target = Variable(data),Variable(target)
        data = data.cuda()
        target = target.cuda()
        pred = resnet18(data)
        loss = criterion(pred,target)
        loss.backward()
        optimizer4.step()
    resnet18.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = resnet18(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(epoch,t,loss.data, correct)