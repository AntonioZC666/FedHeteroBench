from torchvision import transforms
from torch.utils.data import Subset
from utils import mydata
import numpy as np
from utils.femnist_shakes import FEMNIST, ShakeSpeare


def get_dataset(dataset):
    dataset_train, dataset_test = None, None
    if dataset == "mnist":
        from torchvision.datasets import MNIST
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset_train = MNIST("./data/mnist", train=True, transform=train_transform, download=True)
        dataset_test = MNIST("./data/mnist", train=False, transform=test_transform, download=True)

    elif dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        dataset_train = CIFAR10("./data/cifar10", train=True, transform=train_transform, download=True)
        dataset_test = CIFAR10("./data/cifar10", train=False, transform=test_transform, download=True)

    elif dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        dataset_train = mydata.CIFAR100_coarse("./data/cifar100_coarse", train=True, transform=train_transform, download=True)
        dataset_test = mydata.CIFAR100_coarse("./data/cifar100_coarse", train=False, transform=test_transform, download=True)

    elif dataset == "gtsrb":
        from utils.gtsrb import GTSRB
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
        dataset_train = GTSRB("./data/gtsrb", train=True, transform=train_transform)
        dataset_test = GTSRB("./data/gtsrb", train=False, transform=test_transform)

    elif dataset == "tinyimagenet":
        from utils.tinyimagenet import TinyImageNet
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        dataset_train = TinyImageNet("./data/tiny-imagenet-200", train=True, transform=train_transform)
        dataset_test = TinyImageNet("./data/tiny-imagenet-200", train=False, transform=test_transform)
    elif dataset == "svhn":
        from torchvision.datasets import SVHN
        size = 32
        transform = transforms.Compose([
            transforms.Resize([size,size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset_train = SVHN("./data/svhn", split='train', transform=transform, download=True)
        dataset_test = SVHN("./data/svhn", split='test', transform=transform, download=True)
    elif dataset == "femnist":
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图扩展为 3 通道
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ???
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = FEMNIST("./data/femnist", train=True, transform=transform)  # transform???
        dataset_test = FEMNIST("./data/femnist", train=False, transform=transform)  # transform???
    elif dataset == "shakespeare":
        dataset_train = ShakeSpeare("./data/shakespeare", Train=True)  # transform???
        dataset_test = ShakeSpeare("./data/shakespeare", Train=False)  # transform???


    print("Dataset: {}".format(dataset))
    return dataset_train, dataset_test


def get_sub_dataset(dataset, ratio):
    data_length = int(len(dataset) * ratio)
    index = [x for x in range(data_length)]
    # index = np.random.choice(range(len(dataset)), data_length, replace=False)
    dataset_small = Subset(dataset, index)
    return dataset_small


def get_num_classes(dataset_name: str) -> int:
    if dataset_name in ["mnist", "cifar10", "svhn"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "Linnaeus5":
        num_classes = 5
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == "cifar100":
        num_classes = 20
    elif dataset_name == "tinyimagenet":
        num_classes = 200
    elif dataset_name == "imagenet":
        num_classes = 1000
    elif dataset_name == "femnist":
        num_classes = 62
    elif dataset_name == 'shakespeare':
        num_classes == 80
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_input_shape(dataset_name):
    if dataset_name == "cifar10":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "gtsrb":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "mnist":
        input_height = 28
        input_width = 28
        input_channel = 1
    elif dataset_name == "femnist":
        input_height = 28
        input_width = 28
        input_channel = 3  # 1 ?
    elif dataset_name == "celeba":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "cifar100":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "tinyimagenet":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "imagenet":
        input_height = 224
        input_width = 224
        input_channel = 3
    elif dataset_name == "svhn":
        input_height = 32
        input_width = 32
        input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return input_height, input_width, input_channel

