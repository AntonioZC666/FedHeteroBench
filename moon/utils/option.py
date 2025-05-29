import argparse
import torch
from utils.dataset import get_num_classes, get_input_shape


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="ResNet")
    parser.add_argument("--gpu", type=int, default=0)


    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    return args
