import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(32, args.num_classes)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        x = out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        return {'output': out, 'feature_map': x}
