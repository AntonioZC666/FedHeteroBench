import numpy as np
from utils.dataset import get_dataset
from utils.non_iid_distribution import get_distribution

from Server.naive_server import naive_server

from Client.naive_client import naive_client


def get_server_client(args):
    dataset_train, dataset_test = get_dataset(args.dataset)
    # print(dataset_train[0][0])
    if args.dataset == "femnist":
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
    else:
        dict_users = get_distribution(dataset_train, args)
    # print(len(dict_users))
    # print(dict_users)
    # print(type(dict_users), type(dict_users[0]))

    server = None
    client_pool = []

    # FedAvg, 无攻击无防御
    server = naive_server(args, dataset_test)
    for i in range(args.num_users):
        if args.dataset == "femnist":
            client_pool.append(naive_client(args, dataset_train, list(dict_users[i])))
        else:
            client_pool.append(naive_client(args, dataset_train, dict_users[i]))

    return server, client_pool

