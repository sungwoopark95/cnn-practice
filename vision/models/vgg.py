import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import get_cfg

cfg = get_cfg()

class VGG(nn.Module):
    def __init__(self):        
        super(VGG, self).__init__()
        if cfg.dataset == "cifar10":
            num_classes = 10
        elif cfg.dataset == "cifar100":
            num_classes = 100
        
        ## from 4 ~ 2048
        feats = [4, 8, 16, 32, 64, 
                 96, 128, 192, 256, 384, 
                 512, 768, 1024, 1536, 2048]
        
        feat1 = feats[4]
        feat2 = feats[4]
         
        feat3 = feats[6] 
        feat4 = feats[6]
        
        feat5 = feats[8]
        feat6 = feats[8]
        feat7 = feats[8]
        
        feat8 = feats[10]
        feat9 = feats[10]
        feat10 = feats[10]
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = feat1, kernel_size = 3, stride = 1, padding = 'same')
        self.conv2 = nn.Conv2d(in_channels = feat1, out_channels = feat2, kernel_size = 3, padding = 'same')
        
        self.conv3 = nn.Conv2d(in_channels = feat2, out_channels = feat3, kernel_size = 3, padding = 'same')
        self.conv4 = nn.Conv2d(in_channels = feat3, out_channels = feat4, kernel_size = 3, padding = 'same')
        
        self.conv5 = nn.Conv2d(in_channels = feat4, out_channels = feat5, kernel_size = 3, padding = 'same')
        self.conv6 = nn.Conv2d(in_channels = feat5, out_channels = feat6, kernel_size = 3, padding = 'same')
        self.conv7 = nn.Conv2d(in_channels = feat6, out_channels = feat7, kernel_size = 3, padding = 'same')
        
        self.conv8 = nn.Conv2d(in_channels = feat7, out_channels = feat8, kernel_size = 3, padding = 'same')
        self.conv9 = nn.Conv2d(in_channels = feat8, out_channels = feat9, kernel_size = 3, padding = 'same')
        self.conv10 = nn.Conv2d(in_channels = feat9, out_channels = feat10, kernel_size = 3, padding = 'same')
        
        self.num_feat = 7 * 7 * feat10
        
        self.fc1 = nn.Linear(self.num_feat, cfg.head_size1)
        self.fc2 = nn.Linear(cfg.head_size1, cfg.head_size2)
        self.fc3 = nn.Linear(cfg.head_size2, num_classes)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_bn1 = nn.BatchNorm2d(feat1)
        self.conv_bn2 = nn.BatchNorm2d(feat2)
        self.conv_bn3 = nn.BatchNorm2d(feat3)
        self.conv_bn4 = nn.BatchNorm2d(feat4)
        self.conv_bn5 = nn.BatchNorm2d(feat5)
        self.conv_bn6 = nn.BatchNorm2d(feat6)
        self.conv_bn7 = nn.BatchNorm2d(feat7)
        self.conv_bn8 = nn.BatchNorm2d(feat8)
        self.conv_bn9 = nn.BatchNorm2d(feat9)
        self.conv_bn10 = nn.BatchNorm2d(feat10)
        
        self.fc_bn1 = nn.BatchNorm1d(cfg.head_size1)
        self.fc_bn2 = nn.BatchNorm1d(cfg.head_size2)
            
    def forward(self, x):
        ## CNN
        x = self.conv1(x)
        x = F.relu(self.conv_bn1(x))
        x = self.conv2(x)
        x = F.relu(self.conv_bn2(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(self.conv_bn3(x))
        x = self.conv4(x)
        x = F.relu(self.conv_bn4(x))
        x = self.pool(x)
        
        x = self.conv5(x)
        x = F.relu(self.conv_bn5(x))       
        x = self.conv6(x)
        x = F.relu(self.conv_bn6(x))
        x = self.conv7(x)
        x = F.relu(self.conv_bn7(x))
        x = self.pool(x)
        
        x = self.conv8(x)
        x = F.relu(self.conv_bn8(x))       
        x = self.conv9(x)
        x = F.relu(self.conv_bn9(x))
        x = self.conv10(x)
        x = F.relu(self.conv_bn10(x))
        x = self.pool(x)
        
        ## head
        x = x.view(-1, self.num_feat)
        x = self.fc1(x)
        x = F.relu(self.fc_bn1(x))
        x = self.fc2(x)
        x = F.relu(self.fc_bn2(x))
        x = self.fc3(x)
        
        return x