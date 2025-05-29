import copy
from pyexpat import features
import torch
import wandb
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from utils.model import get_model
from collections import deque
import os
os.environ["WANDB_MODE"] = "offline"

class Server:
    def __init__(self, args, benign_dataset):
        self.dataset = args.dataset
        self.device = args.device
        self.num_users = args.num_users
        self.fraction = args.fraction
        self.model = get_model(args)
        self.benign_dataset = benign_dataset
        self.total_sample_bengin = len(self.benign_dataset)
        self.test_loader_benign = DataLoader(self.benign_dataset, batch_size=args.bs, shuffle=True)
        self.local_model_list = None
        self.local_datalen_list = None
        self.use_wandb = args.use_wandb
        self.recent_BA_values = deque(maxlen=10)  # 创建一个最大长度为10的队列
        self.best_BA_values = deque(maxlen=1)
        if args.use_wandb:
            wandb.init(project="federated-learning",
                    #    group=args.defence,
                       group="FedDecorr",
                       job_type=str(args.drichlet_beta),
                       config={**vars(args)},
                       tags=["FedDecorr", args.model, args.dataset, "WAvg"],
                       settings=wandb.Settings(_disable_stats=True))

    # select clients
    def select_clients(self):
        m = max(int(self.fraction * self.num_users), 1)
        clients = np.random.choice(range(self.num_users), m, replace=False)
        print("select clients {} to train".format(clients))
        return clients

    # for aggregate by length of dataset
    def aggregate(self):
        weights = np.array(self.local_datalen_list) / sum(self.local_datalen_list)
        local_models = self.local_model_list
        w_glob = copy.deepcopy(local_models[0])
        n = len(local_models)

        for param in local_models[0]:
            tmp = None
            s = self.model.state_dict()[param].reshape(-1).unsqueeze(0)
            for i in range(n):
                t = local_models[i][param].reshape(-1).unsqueeze(0)
                if tmp is None:
                    tmp = weights[i] * copy.deepcopy(t - s)
                else:
                    tmp = torch.cat((tmp, weights[i] * (t - s)), 0)

            result = torch.sum(tmp, 0)
            w_glob[param] = (result + s).reshape(local_models[0][param].shape)

        self.model.load_state_dict(w_glob)

    # for aggregate
    # def aggregate(self):
    #     local_models = self.local_model_list
    #     w_glob = copy.deepcopy(local_models[0])
    #     n = len(local_models)

    #     for param in local_models[0]:
    #         tmp = None
    #         s = self.model.state_dict()[param].reshape(-1).unsqueeze(0)
    #         for i in range(n):
    #             t = local_models[i][param].reshape(-1).unsqueeze(0)
    #             if tmp is None:
    #                 tmp = copy.deepcopy(t - s)
    #             else:
    #                 tmp = torch.cat((tmp, t - s), 0)

    #         result = torch.sum(tmp, 0)
    #         w_glob[param] = (result / n + s).reshape(local_models[0][param].shape)

    #     self.model.load_state_dict(w_glob)

    # for testing
    def test(self):
        self.model.eval()
        test_accuracy_BA = 0
        test_loss_BA = 0
        criterion = nn.CrossEntropyLoss()

        # For BA
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader_benign):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, features = self.model(images)
                accuracy = (outputs.argmax(1) == labels).sum()
                test_accuracy_BA += accuracy

                loss = criterion(outputs, labels)
                test_loss_BA += loss.item()

        current_BA = test_accuracy_BA / self.total_sample_bengin
        self.recent_BA_values.append(current_BA)
        average_BA = sum(self.recent_BA_values) / len(self.recent_BA_values) # 计算最近10次的平均 BA 值
        if len(self.best_BA_values) == 0 or current_BA > self.best_BA_values[0]:
            self.best_BA_values.append(current_BA) # 保存最高准确率

        if self.use_wandb:
            wandb.log({"BA": test_accuracy_BA / self.total_sample_bengin,
                       "Loss": test_loss_BA / self.total_sample_bengin,
                       "Average BA (last 10)": average_BA,
                       "Best BA": self.best_BA_values[0]})
        # print(test_accuracy_BA, self.total_sample_bengin)
        print("BA: {}".format(current_BA))
        print("Loss: {}".format(test_loss_BA / self.total_sample_bengin))
        print("Average BA (last 10): {}".format(average_BA))
        print("Best BA: {}".format(self.best_BA_values[0]))
        print('=' * 80)

        return test_accuracy_BA / self.total_sample_bengin