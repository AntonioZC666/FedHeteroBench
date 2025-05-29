import os
from datetime import datetime
import copy
import random
from matplotlib.pyplot import cla
import timm
import numpy as np
import timm.utils
import torch
from utils.get_config import get_config
from utils.fix_seed import fix_seed
from utils.get_server_client import get_server_client
from models.MLP import MLP
from tqdm import tqdm

# 计算样本的平均值，生成增强数据
def get_augmentation_proxy_data(args, dataset_train, bs):
    augmentation_data = []
    for i in range(bs):
        indexs = torch.randint(0, len(dataset_train), (10,))
        current_aug_data = torch.zeros_like(dataset_train[0][0])
        current_aug_data = current_aug_data.unsqueeze(0)
        for index in indexs:
            current_aug_data += dataset_train[index][0] / len(indexs)
        augmentation_data.append(current_aug_data.to(args.device))
    return augmentation_data


if __name__ == "__main__":
    args = get_config()
    fix_seed(args.random_seed)

    uda_device = None  # 存储无监督域适应的数据增强样本
    server, client_pool, dataset_train = get_server_client(args)
    uda_device = get_augmentation_proxy_data(args, dataset_train, args.bs)

    if args.save_model:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        save_path = './checkpoints/' + args.dataset + '-' + args.model + '-' + args.rule + str(args.drichlet_beta) + '-' + time_str  # + '-' + str(args.round)
        os.makedirs(save_path, exist_ok=True)
        optimizer = torch.optim.SGD(server.model.parameters(), lr=0, momentum=0, weight_decay=0)
        saver = timm.utils.CheckpointSaver(model=server.model, optimizer=optimizer, args=args, checkpoint_dir=save_path, max_history=1)

    ''' 使用服务端模型作为特征提取器 Phi, 服务端模型最后一层(linear)作为分类器 omega, MLP模型作为投影层 theta
    '''
    classifier = server.model.linear  # 分类器 omega
    # discriminator = MLP(512, 512) # projection layer theta
    for round in tqdm(range(args.round)):
        print('#' * 40 + '  ' + str(round) + '  ' + '#' * 40)
        clients = server.select_clients()
        # print(clients)

        local_models = []
        local_discriminators = []
        for i, idx in enumerate(clients):

            local_model, local_discriminator = client_pool[idx].train(copy.deepcopy(server.model), copy.deepcopy(server.discriminator), uda_device)
            local_models.append(local_model)
            local_discriminators.append(local_discriminator)

        server.local_model_list = local_models
        server.local_disc_list = local_discriminators
        server.aggregate()
        server.aggregate_disc()
        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma
