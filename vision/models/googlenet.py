import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from collections import namedtuple
from cfg import get_cfg

cfg = get_cfg()

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor], 'aux_logits1': Optional[Tensor]}

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True):
        super(GoogLeNet, self).__init__()

        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        elif cfg.dataset == "imagenet":
            num_classes = 1000
            
        self.aux_logits = aux_logits
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding='same')
        )
        
        self.inception1 = Inception(in_channels=192, out1x1=64, to3x3=96, out3x3=128, to5x5=16, out5x5=32, poolout=32)
        self.inception2 = Inception(in_channels=256, out1x1=128, to3x3=128, out3x3=192, to5x5=32, out5x5=96, poolout=64)
        self.inception3 = Inception(in_channels=480, out1x1=192, to3x3=96, out3x3=208, to5x5=16, out5x5=48, poolout=64)
        self.inception4 = Inception(in_channels=512, out1x1=160, to3x3=112, out3x3=224, to5x5=24, out5x5=64, poolout=64)
        self.inception5 = Inception(in_channels=512, out1x1=128, to3x3=128, out3x3=256, to5x5=24, out5x5=64, poolout=64)
        self.inception6 = Inception(in_channels=512, out1x1=112, to3x3=144, out3x3=288, to5x5=32, out5x5=64, poolout=64)
        self.inception7 = Inception(in_channels=528, out1x1=256, to3x3=160, out3x3=320, to5x5=32, out5x5=128, poolout=128)
        self.inception8 = Inception(in_channels=832, out1x1=256, to3x3=160, out3x3=320, to5x5=32, out5x5=128, poolout=128)
        self.inception9 = Inception(in_channels=832, out1x1=384, to3x3=192, out3x3=384, to5x5=48, out5x5=128, poolout=128)
        
        if aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.dropout = nn.Dropout(p=cfg.drop_fc)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool3(x)
        x = self.inception3(x)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
                
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
                
        x = self.inception7(x)
        x = self.maxpool4(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x
        

class Inception(nn.Module):
    def __init__(self, in_channels, out1x1, to3x3, out3x3, to5x5, out5x5, poolout, modified=False):
        super(Inception, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1, eps=0.001),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=to3x3, kernel_size=1),
            nn.BatchNorm2d(to3x3, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(in_channels=to3x3, out_channels=out3x3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out3x3, eps=0.001),
            nn.ReLU()
        )
        
        if modified:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=3, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                nn.ReLU()
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=to5x5, kernel_size=1),
                nn.BatchNorm2d(to5x5, eps=0.001),
                nn.ReLU(),
                nn.Conv2d(in_channels=to5x5, out_channels=out5x5, kernel_size=5, padding='same'),
                nn.BatchNorm2d(out5x5, eps=0.001),
                nn.ReLU()
            )
            
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=in_channels, out_channels=poolout, kernel_size=1),
            nn.BatchNorm2d(poolout, eps=0.001),
            nn.ReLU()            
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
    

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        