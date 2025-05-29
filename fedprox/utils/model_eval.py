import torch
from models.resnet import resnet18
from models.resnet8 import ResNet8
from models.LeNet import *
from models.vgg import *


def get_model(args, pretrain=True):
    model = None

    if args.model == "ResNet":
        if args.ind_dataset in ['cifar10', 'cifar100']:
            model = ResNet8(args).to(args.device)
        else:
            model = resnet18(args).to(args.device)

    if pretrain:
        # save_path = './checkpoints/FedAvg/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta)
        save_path = './checkpoints/Ours/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta) + '-' + str(args.round) + '-' + str(args.wt_increase_epoch) + '-' + str(args.wt_increase_gamma)
        print(save_path)

        # checkpoint = torch.load(save_path + '/model_best.pth.tar')
        # # 检查保存的状态字典的键名
        # print("Saved State Dictionary Keys:")
        # for key in checkpoint['state_dict'].keys():
        #     print(key)

        # # 检查模型状态字典的键名
        # print("Model State Dictionary Keys:")
        # for key in model.state_dict().keys():
        #     print(key)
        # print("*" * 50)

        model.load_state_dict((torch.load(save_path + '/model_best.pth.tar')['state_dict']))

    return model

    # if args.model == "ResNet":
    #     if args.ind_dataset in ['cifar10', 'cifar100']:
    #         model = ResNet18()
    #     else:
    #         model = resnet50(num_classes=args.num_classes, pretrained=True)

    # elif args.model == "DenseNet":
    #     model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=args.num_classes)

    # elif args.model == "WideResNet":
    #     model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=1, droprate=0.0)

    # if pretrain:
    #     save_path = './checkpoints/' + args.ind_dataset + '-' + args.model + '-0'
    #     model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))

    # return model

