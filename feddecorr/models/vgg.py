"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

"""定义不同版本的VGG网络的层配置，每个列表表示一个VGG版本的卷积层和最大池化层的配置
   其中，数字表示卷积层的输出通道数，"64, 64"表示两个64个通道的卷积层；'M'表示一个最大池化层（MaxPooling），用于减小特征图的空间尺寸
"""
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
# 构建VGG网络模型
class VGG(nn.Module):
    """features: 一个 nn.Sequential 对象，包含了VGG网络的卷积层和池化层，用于提取图像特征
       num_class: 分类任务的类别数，默认为100
    """
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features #特征提取。将传入的 features 赋值给 self.features，这部分通常由多个卷积层和池化层组成
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),      # 全连接层，将输入的特征向量（假设特征图的最后一个维度是512）映射到4096维
            nn.ReLU(inplace=True),     # ReLU激活函数用于引入非线性
            nn.Dropout(),              # Dropout层用于防止过拟合，随机丢弃一部分神经元
            nn.Linear(4096, 4096),     # 全连接层，将4096维的特征向量映射到另一个4096维
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class) # 连接层，将4096维的特征向量映射到 num_class 维，即分类任务的任务数
        )
    # 数据的前向传播过程
    def forward(self, x):
        output = self.features(x) # 将输入数据 x 通过特征提取部分 self.features，得到特征图 output
        output = output.view(output.size()[0], -1) # 将特征图 output 展平成二维张量，
                                               # 其中第一维是batch size，第二维是展平后的特征向量，-1表示该维度取决于其它维度大小
        output = self.classifier(output) # 将展平后的特征向量通过分类器部分，得到最终的分类结果 output

        return output

"""根据给定的配置（cfg）生成VGG网络的特征提取部分的层结构
   cfg: 一个列表，包含了VGG网络的层配置
   batch_norm: 一个布尔值，表示是否在卷积层之后添加批归一化层（Batch Normalization）
"""
def make_layers(cfg, batch_norm=False):
    layers = [] # 存储生成的层

    input_channel = 3 # RGB图
    for l in cfg:
        if l == 'M': # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # 向 layers 列表中添加一个 nn.MaxPool2d 层，其 kernel_size 为2，stride 为2
            continue
        # 卷积层
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)] # 输出通道为 l

        if batch_norm:
            layers += [nn.BatchNorm2d(l)] # 向 layers 列表中添加一个 nn.BatchNorm2d 层，其输入通道数为 l

        layers += [nn.ReLU(inplace=True)] # 在列表中添加ReLU激活层，表示在原位操作，节省内存
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(args):
    return VGG(make_layers(cfg['A'], batch_norm=True), args.num_classes)

def vgg13_bn(args):
    return VGG(make_layers(cfg['B'], batch_norm=True), args.num_classes)

def vgg16_bn(args):
    return VGG(make_layers(cfg['D'], batch_norm=True), args.num_classes)

def vgg19_bn(args):
    return VGG(make_layers(cfg['E'], batch_norm=True), args.num_classes)


