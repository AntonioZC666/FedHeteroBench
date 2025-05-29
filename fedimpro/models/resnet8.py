import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from torch.autograd import Variable
from utils_fedimpro.copy import deepcopy


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.feat_length_list = [None for i in range(1)]

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, labels=None, hidden_features=None, hidden_labels=None):
        local_module_i = 0
        feat_list = []
        total_hidden_loss = torch.tensor(0.0)
        extra_feat_index = 0
        real_batch_size = x.size(0)
        next_target = deepcopy(labels)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # out = self.layer2(out)
        x = out # 提取layer2的feature map


        hidden_loss = None
        total_hidden_loss += hidden_loss.item() if hidden_loss is not None else 0.0
        if hidden_features is not None: # fake_features_list非空, 有伪特征
            out, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target,
                                        real_feat_list=feat_list, hidden_loss=hidden_loss,
                                        fake_feat=hidden_features[local_module_i],
                                        fake_label=hidden_labels[local_module_i])
        else:
            out, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target,
                                        real_feat_list=feat_list, hidden_loss=hidden_loss,
                                        fake_feat=None,
                                        fake_label=None)

        self.feat_length_list[local_module_i] = x.size(-1)
        local_module_i += 1
        extra_feat_index += 1


        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        # return {'output': out, 'feature_map': x}
        return out, feat_list



    def process_split_feature(self, real_batch_size, real_feat, real_label, real_feat_list, hidden_loss=None,
                            fake_feat=None, fake_label=None):
        if fake_feat is not None:
            next_feat = torch.cat([real_feat, fake_feat.reshape(-1, real_feat.shape[1], real_feat.shape[2], real_feat.shape[3])], dim=0)
            real_feat_list.append((real_feat.view(real_feat.size(0), -1) * 1.0)[:real_batch_size])
            next_label = torch.cat([real_label, fake_label], dim=0)
        else:
            next_feat = real_feat
            real_feat_list.append(real_feat.view(real_feat.size(0), -1) * 1.0)
            next_label = real_label

        return next_feat, next_label


def ResNet8(args):
    return ResNet(BasicBlock, [1, 1, 1], args.num_classes)