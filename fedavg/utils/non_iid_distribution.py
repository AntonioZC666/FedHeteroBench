import numpy as np


def get_distribution(dataset_train, args):
    if args.rule == "iid":
        num_items = int(len(dataset_train) / args.num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset_train))]
        for i in range(args.num_users):
            dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
        print(dict_users)
        return dict_users
    elif args.rule == "pathological":
        pass
    elif args.rule == "drichlet":
        min_size = 0
        min_require_size = 10
        K = args.num_classes
        if args.dataset == 'svhn':  # or args.dataset == 'femnist':
            y_train = np.array(dataset_train.labels)
        else:
            y_train = np.array(dataset_train.targets)
        N = len(dataset_train)
        dict_users = {}

        idx_batch = None
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.num_users)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.drichlet_beta, args.num_users))
                proportions = np.array(
                    [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.num_users):
            # np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]

        return dict_users
