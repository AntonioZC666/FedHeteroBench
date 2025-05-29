import numpy as np
from utils.dataset import get_dataset
from utils.non_iid_distribution import get_distribution

from Server.naive_server import naive_server

from Client.naive_client import naive_client


def get_server_client(args):
    dataset_train, dataset_test = get_dataset(args.dataset)
    # print(dataset_train[0][0])
    dict_users = get_distribution(dataset_train, args)
    # print(len(dict_users))
    # print(dict_users)

    server = None
    client_pool = []

    # FedAvg, 无攻击无防御
    server = naive_server(args, dataset_test)
    for i in range(args.num_users):
        client_pool.append(naive_client(args, dataset_train, dict_users[i]))

    return server, client_pool

