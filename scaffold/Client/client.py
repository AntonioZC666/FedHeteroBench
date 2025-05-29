import time
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
    def train(self, net, control_local, control_global): # 全局模型 x, 本地控制变量 Ci, 全局控制变量 C
        # 将全局模型 x 赋值给本地模型 yi

        global_weights = net.state_dict()  # x

        net.train()  # net即为 yi

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        control_global_w = control_global.state_dict() # C
        control_local_w = control_local.state_dict()  # Ci

        count = 0
        for iter in range(local_ep): # local_cp即为论文中的 K local_update ???，lr即为 local step-size : nl
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward() # line 9, 计算梯度 gi
                optimizer.step()  # yi = yi - nl * gi

                local_weights = net.state_dict()
                for w in local_weights:
                    # line 10 in algo, yi = yi - nl * (C - Ci)
                    local_weights[w] = local_weights[w] - lr * (control_global_w[w]-control_local_w[w])

                # update local model params
                net.load_state_dict(local_weights)

                count += 1

        new_control_local_w = control_local.state_dict()  # 新的本地控制变量 Ci+
        control_delta = copy.deepcopy(control_local_w)   # 本地控制变量的差值 delta_Ci
        # model_weights -> y_(i)
        model_weights = net.state_dict()  # 本地模型的拷贝
        local_delta = copy.deepcopy(model_weights)  # 本地模型的差值 delta_yi

        for w in model_weights:
            #line 12 in algo, option 2.  Ci - C + (x - yi) / (K * nl)
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
            #line 13
            control_delta[w] = new_control_local_w[w] - control_local_w[w]  # delta_Ci = Ci+ - Ci
            local_delta[w] -= global_weights[w]  #???  delta_yi = yi - x

            # new_control_local_w[w] = control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
            # local_delta[w] = model_weights[w] - global_weights[w]
            # control_delta[w] = new_control_local_w[w] - control_local_w[w]

        # update new control_local model
        # control_local.load_state_dict(new_control_local_w)  # 赋值, Ci = Ci+

        return net.state_dict(), control_delta, local_delta, new_control_local_w, self.data_length  # 返回client模型, delta_Ci, delta_yi, Ci+