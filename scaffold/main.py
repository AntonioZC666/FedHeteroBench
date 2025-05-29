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
from utils.model import get_model
from tqdm import tqdm

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

    # server.model即为全局模型 x ; control_global即为全局控制变量 C ; 全局步长 ng 论文中设为 1
    control_global = get_model(args)

    server.model.train()
    control_weights = control_global.state_dict()

    # model for local control varietes, 为每个客户端创建本地控制变量
    local_controls = [get_model(args) for i in range(args.num_users)]
    for net in local_controls:
        net.load_state_dict(control_weights)  # 初始化本地控制变量 Ci，与全局控制变量一样

    # devices that participate (sample size)
    m = max(int(args.fraction * args.num_users), 1)

    # initiliase total delta to 0 (sum of all control_delta, triangle Ci)
    delta_c = copy.deepcopy(server.model.state_dict())
    # sum of delta_y / sample size
    delta_x = copy.deepcopy(server.model.state_dict())

    for round in tqdm(range(args.round)):
        print('#' * 40 + '  ' + str(round) + '  ' + '#' * 40)

        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_x:
            delta_x[ci] = 0.0

        server.model.train()

        # line 4, clients即为算法中的 S
        clients = server.select_clients()
        local_models = []
        local_data_length = []
        for i, idx in enumerate(clients):
            # line 6-15, 将全局模型 x, 本地控制变量 Ci,全局控制变量 C 传递给 client; 返回客户端本地模型 yi, delta_Ci, delta_yi, Ci+
            local_model, local_delta_c, local_delta, control_local_w, data_len = client_pool[idx].train(copy.deepcopy(server.model), local_controls[idx], control_global)
            local_models.append(local_model)
            local_data_length.append(data_len)

            if round != 0:
                local_controls[idx].load_state_dict(control_local_w)

            # line16
            for w in delta_c:
                if round == 0:
                    delta_x[w] += local_model[w]  # ???
                    # delta_x[w] += local_delta[w]
                else:
                    delta_x[w] += local_delta[w]
                    delta_c[w] += local_delta_c[w]

        #update the delta C (line 16)
        for w in delta_c:
            delta_c[w] /= m
            delta_x[w] /= m

        # update global control variate (line17)
        control_global_W = control_global.state_dict()  # 全局控制变量 C
        global_weights = server.model.state_dict()  # 全局模型 x
        # equation taking Ng, global step size = 1
        for w in control_global_W:
            # control_global_W[w] += delta_c[w]
            if round == 0:
                global_weights[w] = delta_x[w]
            else:
                # print(global_weights[w])
                # print(delta_x[w])
                global_weights[w] += delta_x[w].to(dtype=global_weights[w].dtype)
                control_global_W[w] += ((m / args.num_users) * delta_c[w]).to(dtype=control_global_W[w].dtype)


        # update global model
        control_global.load_state_dict(control_global_W)
        server.model.load_state_dict(global_weights)

        #########scaffold algo complete##################


        server.local_model_list = local_models
        server.local_datalen_list = local_data_length
        server.aggregate()
        accuracy = server.test()

        if args.save_model:
            saver.save_checkpoint(round, metric=accuracy) # 保存准确率最高的权重

        # lr_decay
        if (round + 1) % args.lr_decay_epoch == 0:
            args.lr = args.lr * args.lr_decay_gamma
