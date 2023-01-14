import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class ResNet(nn.Module):
    def __init__(self):        
        super(ResNet, self).__init__()
        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        elif cfg.dataset == "imagenet":
            num_classes = 1000
        
        ## from 4 ~ 2048
        feats = [4, 8, 16, 32, 64, 
                 96, 128, 192, 256, 384, 
                 512, 768, 1024, 1536, 2048]
        
        feat1 = feats[4]
        feat2, feat3, feat4, feat5, feat6, feat7, feat8 = [feats[4] for _ in range(2, 8+1)] # 64
        feat9, feat10, feat11, feat12, feat13, feat14, feat15, feat16 = [feats[6] for _ in range(9, 16+1)] # 128
        feat17, feat18, feat19, feat20, feat21, feat22, feat23, feat24, feat25, feat26 = [feats[8] for _ in range(17, 26+1)] # 256
        feat27, feat28, feat29, feat30, feat31, feat32 = [feats[10] for _ in range(27, 32+1)] # 512
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=feat1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=feat1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res1 = ResBlock(in_channels=feat2, intermediate=feat3, out_channels=feat4)
        self.res2 = ResBlock(in_channels=feat4, intermediate=feat5, out_channels=feat6)
        self.res3 = ResBlock(in_channels=feat6, intermediate=feat7, out_channels=feat8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res4 = ResBlock(in_channels=feat8, intermediate=feat9, out_channels=feat10)
        self.res5 = ResBlock(in_channels=feat10, intermediate=feat11, out_channels=feat12)
        self.res6 = ResBlock(in_channels=feat12, intermediate=feat13, out_channels=feat14)
        self.res7 = ResBlock(in_channels=feat14, intermediate=feat15, out_channels=feat16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res8 = ResBlock(in_channels=feat16, intermediate=feat17, out_channels=feat18)
        self.res9 = ResBlock(in_channels=feat18, intermediate=feat19, out_channels=feat20)
        self.res10 = ResBlock(in_channels=feat20, intermediate=feat21, out_channels=feat22)
        self.res11 = ResBlock(in_channels=feat22, intermediate=feat23, out_channels=feat24)
        self.res12 = ResBlock(in_channels=feat24, intermediate=feat25, out_channels=feat26)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res13 = ResBlock(in_channels=feat26, intermediate=feat27, out_channels=feat28)
        self.res14 = ResBlock(in_channels=feat28, intermediate=feat29, out_channels=feat30)
        self.res15 = ResBlock(in_channels=feat30, intermediate=feat31, out_channels=feat32)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=cfg.drop_fc)
        self.fc1 = nn.Linear(feat32, out_features=cfg.head_fc1)
        self.fc2 = nn.Linear(cfg.head_fc1, out_features=num_classes)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool2(x)
        
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.pool3(x)
        
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.pool4(x)
        
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.pool5(x)
        
        ## head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

class PlainBlock(nn.Module):
    def __init__(self, in_channels, intermediate, out_channels):
        super(PlainBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=intermediate),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, intermediate, out_channels):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=intermediate),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
        self.downsample = nn.Sequential(
            nn.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        branch = self.block(x)
        identity = self.downsample(x)
        x = branch + identity
        x = F.relu(x)
        return x