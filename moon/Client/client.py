import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.model import get_model


class Client:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        self.data_length = len(self.data_index) # length of dataset
        self.temperature = args.temperature
        self.contrastive_alpha = args.contrastive_alpha

    # for training
    def train(self, net, old_models):
        global_net = get_model(self.args)
        global_net.load_state_dict(net.state_dict())

        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        cos=torch.nn.CosineSimilarity(dim=-1)
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for iter in range(local_ep):
            # epoch_loss_collector = []
            # epoch_loss1_collector = []
            # epoch_loss2_collector = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                _, pro1, outputs = net(images)  # line 14. TODO modify model
                if outputs.dim() == 1:  # 若降维成1维，扩充成[1,num_classes]
                    outputs = outputs.unsqueeze(0)
                predictive_loss = criterion(outputs, labels)

                _, pro2, _ = global_net(images) # line 15

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in old_models:
                    _, pro3, _ = previous_net(images)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= self.temperature
                targets = torch.zeros(images.size(0)).to(self.args.device).long()
                # print("logits shape: {}".format(logits.shape))
                # print("targets shape: {}".format(targets.shape))

                contrastive_loss = self.contrastive_alpha * criterion(logits, targets)

                loss = predictive_loss + contrastive_loss
                loss.backward()
                optimizer.step()

                # record loss
            #     epoch_loss_collector.append(loss.item())
            #     epoch_loss1_collector.append(predictive_loss.item())
            #     epoch_loss2_collector.append(contrastive_loss.item())

            # epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            # epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            # epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            # print("loss: {}, predictive_loss: {}, contrastive_loss: {}".format(epoch_loss, epoch_loss1, epoch_loss2))

        return net.state_dict(), self.data_length