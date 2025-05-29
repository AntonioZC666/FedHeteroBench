import os
import math
from datetime import datetime
import copy
import random
import timm
import numpy as np
import timm.utils
import torch
from utils.get_config import get_config
from utils.fix_seed import fix_seed
from utils.get_server_client import get_server_client
from tqdm import tqdm
from models.npn import NPN
from utils.model import get_model

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)

def merge_ep_prior_and_get_cavity(client_m, client_s):
    lmd = [(1/(s+EPS)) for s in client_s]
    theta_s = 1/(sum(lmd)+EPS)
    theta_m = theta_s * sum([m*l for m,l in zip(client_m, lmd)])
    theta_m = torch.clamp(theta_m, EPS, 5)
    theta_s = torch.clamp(theta_s, EPS, 5)
    cavities = []
    for i in range(len(client_m)):
        rest_lmd = lmd[:i] + lmd[i+1:]
        rest_m = client_m[:i] + client_m[i+1:]
        c_s = (1 / (sum(rest_lmd) + EPS))
        c_m = (c_s * sum([m*l for m,l in zip(rest_m, rest_lmd)]))
        cavities.append((c_m, c_s))

    return theta_m, theta_s, cavities


if __name__ == "__main__":
    args = get_config()
    fix_seed(args.random_seed)
    server, client_pool = get_server_client(args)

    npn_models = [NPN(args, get_model(args)) for i in range(args.num_users)]

    cavities_list = [(torch.zeros(10).cuda(), torch.ones(10).cuda() / (args.num_users - 1)) for i in range(args.num_users)]  # TODO ???

    if args.save_model:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        save_path = './checkpoints/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta) + '-' + time_str  # + '-' + str(args.round)
        os.makedirs(save_path, exist_ok=True)
        optimizer = torch.optim.SGD(server.model.parameters(), lr=0, momentum=0, weight_decay=0)
        saver = timm.utils.CheckpointSaver(model=server.model, optimizer=optimizer, args=args, checkpoint_dir=save_path, max_history=1)

    for round in tqdm(range(args.round)):
        print('#' * 40 + '  ' + str(round) + '  ' + '#' * 40)
        clients = server.select_clients()

        global_w_next = copy.deepcopy(server.model.state_dict())
        for key in global_w_next:
            global_w_next[key] = 0

        # local_models = []
        clients_mu, clients_sigma = [], []
        local_data_length = []
        for i, idx in enumerate(clients):

            local_model, mu, sigma, data_len = client_pool[idx].train(copy.deepcopy(server.model), npn_models[idx], cavities_list[idx])
            # local_models.append(local_model)
            clients_mu.append(mu)
            clients_sigma.append(sigma)
            local_data_length.append(data_len)
            # print("out of client --- mu: {} , sigma: {}".format(mu, sigma))

            for key in global_w_next:
                global_w_next[key] += local_model[key] / 10   # args.K 客户端数量. (args.fraction * args.num_users)

        # # server.local_model_list = local_models
        server.local_datalen_list = local_data_length
        server.model.load_state_dict(global_w_next)
        _, _, cavities = merge_ep_prior_and_get_cavity(clients_mu, clients_sigma)  # 被选中客户端的cavities
        # print("cavities: {}".format(cavities))
        for i, idx in enumerate(clients): # 更新全局
            cavities_list[idx] = cavities[i]

        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma
