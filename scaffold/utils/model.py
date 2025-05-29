from models.LeNet import LeNet
from models.CNNMnist import CNNMnist
from models.resnet import resnet18
from models.resnet8 import ResNet8
from models.vgg import vgg11_bn
from models.mobilenet import MobileNet


def get_model(args):
    model = None

    if args.model == "LeNet":
        model = LeNet(args).to(args.device)
    elif args.model == "ResNet":
        model = ResNet8(args).to(args.device)
        # model = resnet18(args).to(args.device)
    elif args.model == "vgg":
        model = vgg11_bn(args).to(args.device)
    elif args.model == "MobileNet":
        model = MobileNet(args).to(args.device)

    # print("Model: {}".format(args.model))

    return model
