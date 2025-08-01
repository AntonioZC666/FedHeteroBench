import os
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
from scipy import special

def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]

    if metric=='cosine':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric=='only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores>0.9, 0.01, float('inf'))
    elif metric=='l1':
        difference = np.linalg.norm(local_distributions-hypo_distribution, ord=1, axis=1)
    elif metric=='l2':
        difference = np.linalg.norm(local_distributions-hypo_distribution, axis=1)
    elif metric=='kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference

if __name__ == "__main__":
    args = get_config()
    fix_seed(args.random_seed)
    server, client_pool = get_server_client(args)

    if args.save_model:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        save_path = './checkpoints/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta) + '-' + time_str  # + '-' + str(args.round)
        os.makedirs(save_path, exist_ok=True)
        optimizer = torch.optim.SGD(server.model.parameters(), lr=0, momentum=0, weight_decay=0)
        saver = timm.utils.CheckpointSaver(model=server.model, optimizer=optimizer, args=args, checkpoint_dir=save_path, max_history=1)

    client_class_count = []
    for i in range(len(client_pool)):
        client_class_count.append(client_pool[i].class_distribution)
    client_class_count = np.array(client_class_count)
    # print(type(client_class_count), client_class_count)
    global_dist = np.ones(client_class_count.shape[1]) / client_class_count.shape[1]
    # print(client_class_count)
    # print(global_dist)

    for round in tqdm(range(args.round)):
        print('#' * 40 + '  ' + str(round) + '  ' + '#' * 40)
        clients = server.select_clients()
        # print(clients)

        difference = get_distribution_difference(client_class_count, clients, metric='kl', hypo_distribution=global_dist)

        local_models = []
        local_data_length = []
        for i, idx in enumerate(clients):

            local_model, data_len = client_pool[idx].train(copy.deepcopy(server.model))
            local_models.append(local_model)
            local_data_length.append(data_len)

        server.local_model_list = local_models
        server.local_datalen_list = local_data_length
        server.aggregate(difference)
        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma
