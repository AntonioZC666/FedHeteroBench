import torch
from torch import nn
import torch.nn.functional as F
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

        self.if_updated = True
        self.all_global_unlabeled_z = None
        self.unlabeled = None
        self.minibatches = []
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            self.minibatches.append((images, labels))

    # for training
    def train(self, net, discriminator, unlabeled=None): # net == featurizer  , classifier
        self.unlabeled = unlabeled
        net.train()
        discriminator.train()
        discriminator.to(self.args.device)
        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()

        # classifier = net.linear

        mu = 0.5
        lam = 1.0
        gamma = 1.0
        tau1 = 2.0
        tau2 = 2.0  # 0.5
        # 判别器的优化器
        disc_opt = torch.optim.SGD(discriminator.parameters(),
                                    lr=0.01,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        # 生成器的优化器
        gen_opt = torch.optim.SGD(net.parameters(),  # (list(net.parameters()) + list(net.parameters())),
                                    lr=0.01,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        original_feature = copy.deepcopy(net) # 保存原始特征提取器的副本
        # original_classifier = copy.deepcopy(classifier)  # 保存原始分类器的副本

        ''' test '''
        for iter in range(local_ep): # test4
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                all_y = F.one_hot(labels, self.args.num_classes).to(self.args.device)
                all_unlabeled = torch.cat([x for x in self.unlabeled]).to(self.args.device)  # 伪数据
                all_unlabeled = all_unlabeled[:len(images)]  # 将大小调整到与当前数据大小一致

                q = torch.ones((len(all_y), self.args.num_classes)).to(self.args.device) / self.args.num_classes
                # print(len(images), len(all_y), len(all_unlabeled), len(q))
                all_unlabeled, q = self.get_unlabeled_by_self(images, all_y, all_unlabeled, 1)  # 伪图片，伪标签

                # 若模型已更新
                if self.if_updated:
                    original_feature = copy.deepcopy(net) # 保存原始的特征提取器(全局特征提取器)
                    # original_classifier = copy.deepcopy(classifier) # 保存原始的分类器
                    self.all_global_unlabeled_z = original_feature(all_unlabeled)[1].clone().detach()  # Phi_g(x_p)
                    self.if_updated = False

                output_x, all_self_z = net(images)  # y, Phi_i(x)
                output_xp, all_unlabeled_z = net(all_unlabeled)  # y_p, Phi_i(x_p)

                # Max Step: 计算判别器的对比损失(流程图右半部分)
                embedding1 = discriminator(all_unlabeled_z.clone().detach())
                embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)].clone().detach()) # 将大小调整到与当前数据大小一致  .clone().detach()
                embedding3 = discriminator(all_self_z.clone().detach())
                # 最大化对比损失(* 变 /，正 变 负)
                # disc_loss = torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
                disc_loss = - torch.log(torch.exp(self.sim(embedding1, embedding2) / tau1) / (torch.exp(self.sim(embedding1, embedding2) / tau1) + torch.exp(self.sim(embedding1, embedding3) / tau2)))

                disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)  # 对比损失(loss第三项)
                print("disc_loss1: {}".format(disc_loss))
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step() # 更新判别器模型

                # Min Step: 训练生成器
                embedding1 = discriminator(all_unlabeled_z)
                embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)]) # 将大小调整到与当前数据大小一致
                embedding3 = discriminator(all_self_z)
                # 最小化对比损失(* 变 /，负 变 正)
                # disc_loss = - torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
                disc_loss = torch.log(torch.exp(self.sim(embedding1, embedding2) / tau1) / (torch.exp(self.sim(embedding1, embedding2) / tau1) + torch.exp(self.sim(embedding1, embedding3) / tau2)))

                disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)  # 对比损失(loss第三项)

                classifier_loss = criterion(output_x, labels)  # 计算分类器的原数据损失(loss第一项)
                aug_penalty = criterion(output_xp, q)  # 计算分类器的伪数据损失(loss第二项)

                gen_loss =  classifier_loss + (mu * disc_loss) + lam * aug_penalty
                print("gen_loss: {} \tcls_loss: {} \taug_penalty: {} \tdisc_loss2: {}".format(gen_loss, classifier_loss, aug_penalty, disc_loss))
                # disc_opt.zero_grad()
                gen_opt.zero_grad()
                # disc_loss.backward()
                gen_loss.backward()
                # disc_opt.step()
                gen_opt.step()




        # for iter in range(local_ep):
        #     classifier = net.linear # test3
        #     for batch_idx, (images, labels) in enumerate(self.train_loader):
        #         # classifier = net.linear # test
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         all_y = F.one_hot(labels, self.args.num_classes).to(self.args.device)
        #         all_unlabeled = torch.cat([x for x in self.unlabeled]).to(self.args.device)
        #         all_unlabeled = all_unlabeled[:len(images)]  # 将大小调整到与当前数据大小一致

        #         q = torch.ones((len(all_y), self.args.num_classes)).to(self.args.device) / self.args.num_classes
        #         # print(len(images), len(all_y), len(all_unlabeled), len(q))
        #         all_unlabeled, q = self.get_unlabeled_by_self(images, all_y, all_unlabeled, 1)

        #         # 若模型已更新
        #         if self.if_updated:
        #             original_feature = copy.deepcopy(net) # 保存原始的特征提取器(全局特征提取器)
        #             original_classifier = copy.deepcopy(classifier) # 保存原始的分类器
        #             self.all_global_unlabeled_z = original_feature(all_unlabeled)[1].clone().detach()
        #             self.if_updated = False

        #         # 计算未标记数据和已标记数据的特征
        #         _, all_unlabeled_z = net(all_unlabeled)
        #         _, all_self_z = net(images)
        #         all_unlabeled_z = all_unlabeled_z.to(self.args.device)
        #         all_self_z = all_self_z.to(self.args.device)

        #         # 计算判别器的对比损失
        #         embedding1 = discriminator(all_unlabeled_z.clone().detach())
        #         embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)]) # 将大小调整到与当前数据大小一致
        #         embedding3 = discriminator(all_self_z.clone().detach())
        #         # print(embedding1.shape, embedding2.shape, embedding3.shape)

        #         disc_loss = torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))

        #         disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)

        #         disc_opt.zero_grad()
        #         disc_loss.backward()
        #         disc_opt.step() # 更新模型参数以减少损失


        #         embedding1 = discriminator(all_unlabeled_z)
        #         embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)])
        #         embedding3 = discriminator(all_self_z)

        #         disc_loss = - torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
        #         disc_loss = torch.sum(disc_loss) / len(embedding1)

        #         # 计算分类器的损失
        #         all_preds = classifier(all_self_z)
        #         classifier_loss = - torch.mean(torch.sum(F.log_softmax(all_preds, 1) * all_y, 1))
        #         # 计算数据增强的惩罚项
        #         aug_penalty = - torch.mean(torch.sum(torch.mul(F.log_softmax(classifier(all_unlabeled_z), 1), q), 1))
        #         # 计算生成器的总损失
        #         gen_loss =  classifier_loss + (mu * disc_loss) + lam * aug_penalty

        #         disc_opt.zero_grad()
        #         gen_opt.zero_grad()
        #         gen_loss.backward()
        #         gen_opt.step()


        # for batch_idx, (images, labels) in enumerate(self.train_loader):
        #     images, labels = images.to(self.args.device), labels.to(self.args.device)
        #     all_y = F.one_hot(labels, self.args.num_classes).to(self.args.device)
        #     all_unlabeled = torch.cat([x for x in self.unlabeled]).to(self.args.device)
        #     all_unlabeled = all_unlabeled[:len(images)]  # 将大小调整到与当前数据大小一致

        #     q = torch.ones((len(all_y), self.args.num_classes)).to(self.args.device) / self.args.num_classes
        #     # print(len(images), len(all_y), len(all_unlabeled), len(q))
        #     all_unlabeled, q = self.get_unlabeled_by_self(images, all_y, all_unlabeled, 1)

        #     # 若模型已更新
        #     if self.if_updated:
        #         original_feature = copy.deepcopy(net) # 保存原始的特征提取器
        #         original_classifier = copy.deepcopy(classifier) # 保存原始的分类器
        #         self.all_global_unlabeled_z = original_feature(all_unlabeled)[1].clone().detach()
        #         self.if_updated = False

        #     # 计算未标记数据和已标记数据的特征
        #     _, all_unlabeled_z = net(all_unlabeled)
        #     _, all_self_z = net(images)
        #     all_unlabeled_z = all_unlabeled_z.to(self.args.device)
        #     all_self_z = all_self_z.to(self.args.device)

        #     # 计算判别器的对比损失
        #     embedding1 = discriminator(all_unlabeled_z.clone().detach())
        #     embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)]) # 将大小调整到与当前数据大小一致
        #     embedding3 = discriminator(all_self_z.clone().detach())
        #     # print(embedding1.shape, embedding2.shape, embedding3.shape)

        #     disc_loss = torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))

        #     disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)

        #     disc_opt.zero_grad()
        #     disc_loss.backward()
        #     disc_opt.step() # 更新模型参数以减少损失


        #     embedding1 = discriminator(all_unlabeled_z)
        #     embedding2 = discriminator(self.all_global_unlabeled_z[:len(images)])
        #     embedding3 = discriminator(all_self_z)

        #     disc_loss = - torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
        #     disc_loss = torch.sum(disc_loss) / len(embedding1)

        #     # 计算分类器的损失
        #     all_preds = classifier(all_self_z)
        #     classifier_loss = - torch.mean(torch.sum(F.log_softmax(all_preds, 1) * all_y, 1))
        #     # 计算数据增强的惩罚项
        #     aug_penalty = - torch.mean(torch.sum(torch.mul(F.log_softmax(classifier(all_unlabeled_z), 1), q), 1))
        #     # 计算生成器的总损失
        #     gen_loss =  classifier_loss + (mu * disc_loss) + lam * aug_penalty

        #     disc_opt.zero_grad()
        #     gen_opt.zero_grad()
        #     gen_loss.backward()
        #     gen_opt.step()


        return net.state_dict(), discriminator.state_dict()


    def get_unlabeled_by_self(self, all_x, all_y, all_unlabeled, K=1): # 根据已有数据生成新的无标签数据
        # one_hot_all_y = F.one_hot(all_y, self.num_classes).to(all_x.device)
        one_hot_all_y = all_y
        new_q = torch.ones((len(all_y), self.args.num_classes)).to(self.args.device) / self.args.num_classes
        for i in range(len(all_unlabeled)): # all_unlabeled
            for j in range(K):
                index = torch.randint(0, len(all_x), (1,))
                # if all_y[index] == all_y[i % len(all_y)]:
                #     index = torch.randint(0, len(all_x), (1,))
                all_unlabeled[i] = all_unlabeled[i] + all_x[index]
                new_q[i] = new_q[i] + one_hot_all_y[index]
            all_unlabeled[i] = all_unlabeled[i] / (K + 1)
            new_q[i] = new_q[i] / (K + 1)

        return all_unlabeled, new_q


    def sim(self, x1, x2): # 计算两个张量之间的余弦相似度
        return torch.cosine_similarity(x1, x2, dim=1)