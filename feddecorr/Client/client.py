import torch
from torch import nn
from torch.utils.data import DataLoader
from feddecorr import FedDecorrLoss


class Client:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        self.data_length = len(self.data_index) # length of dataset

    # for training
    def train(self, net):

        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        feddecorr = FedDecorrLoss()
        for iter in range(local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs, features = net(images)
                loss = criterion(outputs, labels)
                loss_feddecorr = feddecorr(features)
                loss = loss + self.args.feddecorr_coef * loss_feddecorr
                # print(loss)
                loss.backward()
                optimizer.step()

        return net.state_dict(), self.data_length