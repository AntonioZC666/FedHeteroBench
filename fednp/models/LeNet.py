import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # 32 * 32 * 3 -> 28 * 28 * 6 -> 14 * 14 * 6
        x = self.pool(F.relu(self.conv1(x)))
        # 14 * 14 * 6 -> 10 * 10 * 16 -> 5 * 5 * 16
        x = self.pool(F.relu(self.conv2(x)))
        out = x
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return {'output': x, 'feature_map': out}