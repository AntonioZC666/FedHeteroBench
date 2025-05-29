import os
from datetime import datetime
import copy
import random
from threading import local
import timm
import numpy as np
import timm.utils
import torch
from utils.get_config import get_config
from utils.fix_seed import fix_seed
from utils.get_server_client import get_server_client
from tqdm import tqdm

def generate_layer_generator_dict(args, model, layer_generator, x, labels=None):
        # Server is responsible to call this function.
        model.eval()
        model.to(args.device)
        layer_generator.to(args.device)
        x = x.to(args.device)
        with torch.no_grad():
            output, feat_list = model(x, labels)
            for i, feat in enumerate(feat_list):
                layer_generator.initial_model_params(feat.to("cpu"), model.feat_length_list[i])


if __name__ == "__main__":
    args = get_config()
    fix_seed(args.random_seed)
    server, client_pool = get_server_client(args)

    fake_data = torch.randn(args.bs, args.input_channel, args.input_height, args.input_width)
    generate_layer_generator_dict(args, server.model, server.layer_generator, fake_data) # 初始化layer_generator

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
        local_models = []
        local_data_length = []
        local_layer_generators = []
        for i, idx in enumerate(clients):

            local_model, local_layer_generator, data_len = client_pool[idx].train(copy.deepcopy(server.model), copy.deepcopy(server.layer_generator), round)
            local_models.append(local_model)
            local_data_length.append(data_len)
            local_layer_generators.append(local_layer_generator)

        server.local_model_list = local_models
        server.local_datalen_list = local_data_length
        server.layer_generator_list = local_layer_generators
        server.aggregate()
        server.aggregate_layer_generator()
        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma

