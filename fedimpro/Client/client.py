import torch
from torch import nn
from torch.utils.data import DataLoader
import copy


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
    def train(self, net, layer_generator, round):
        net = net.to(self.args.device)
        layer_generator = layer_generator.to(self.args.device)

        net.train()

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

                real_batch_size = images.size(0)
                split_loss = 0
                fake_features_list = []
                fake_labels_list = []
                fake_features, fake_labels = layer_generator.sample(images, labels)
                fake_features_list.append(fake_features.detach())
                fake_labels_list.append(fake_labels.detach())

                outputs, feat_list = net(images, labels, fake_features_list, fake_labels_list)
                for i, feat in enumerate(feat_list):
                    cache_feature_means = copy.deepcopy(layer_generator.feature_synthesis.feature_means)
                    decode_error = layer_generator.update(feat.detach(), labels.detach())
                    loss = torch.tensor(0.0)
                    split_loss += loss
                    diff = layer_generator.feature_synthesis.feature_means - cache_feature_means

                all_labels = [labels]
                for fake_labels in fake_labels_list:
                    all_labels.append(fake_labels)
                all_labels = torch.cat(all_labels, dim=0)

                fake_weight = 1.0 * (round / 500)  # 全局通信轮次
                loss_real = criterion(outputs[:real_batch_size], all_labels[:real_batch_size])
                loss_fake = criterion(outputs[real_batch_size:], all_labels[real_batch_size:])
                loss = loss_real * 1 + loss_fake * fake_weight
                total_loss = split_loss + loss
                total_loss.backward()


                optimizer.step()

        return net.state_dict(), layer_generator, self.data_length # layer_generator.state_dict()