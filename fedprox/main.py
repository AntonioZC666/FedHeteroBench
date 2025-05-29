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

if __name__ == "__main__":
    args = get_config()
    fix_seed(args.random_seed)
    server, client_pool = get_server_client(args)

    if args.save_model:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        save_path = './checkpoints/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta) + '-' + str(args.mu) + '-' + time_str  # + '-' + str(args.round)
        os.makedirs(save_path, exist_ok=True)
        optimizer = torch.optim.SGD(server.model.parameters(), lr=0, momentum=0, weight_decay=0)
        saver = timm.utils.CheckpointSaver(model=server.model, optimizer=optimizer, args=args, checkpoint_dir=save_path, max_history=1)

    for round in tqdm(range(args.round)):
        print('#' * 40 + '  ' + str(round) + '  ' + '#' * 40)
        clients = server.select_clients()
        local_models = []
        local_data_length = []
        for i, idx in enumerate(clients):

            local_model, data_len = client_pool[idx].train(copy.deepcopy(server.model), server)
            local_models.append(local_model)
            local_data_length.append(data_len)

        server.local_model_list = local_models
        server.local_datalen_list = local_data_length
        server.aggregate()
        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma
