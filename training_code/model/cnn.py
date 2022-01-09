import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from utils.options import args



device = torch.device(f"cuda:{args.gpus[0]}")



class CNN(nn.Module):

    def __init__(self, num_classes=20):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.do2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.do3 = nn.Dropout(p=0.1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        # self.conv5 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.do4 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(16*16*16, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # if not (args.test_only or args.inference_only):
        x = self.do2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        # if not (args.test_only or args.inference_only):
        x = self.do3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        # x = self.bn5(self.conv5(x))
        x = self.maxpool(x)
    
        x = torch.flatten(x, 1)
        # # if not (args.test_only or args.inference_only):
        x = self.do4(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x

