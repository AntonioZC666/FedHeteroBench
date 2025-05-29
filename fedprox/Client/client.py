import copy
import torch
from torch import nn
from torch.utils.data import DataLoader


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
    def train(self, net, server):

        net.train()
        global_model = copy.deepcopy(server.model)

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for iter in range(local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs = net(images)

                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(net.parameters(), global_model.parameters()): # 同时遍历本地模型和全局模型，计算两个参数的欧氏距离
                    proximal_term += (w - w_t).norm(2)

                loss = criterion(outputs, labels) + self.args.mu * proximal_term   # (self.args.mu / 2)
                loss.backward()
                optimizer.step()

        return net.state_dict(), self.data_length