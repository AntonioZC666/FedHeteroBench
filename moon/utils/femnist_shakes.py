import json
import os
from PIL import Image
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.datasets import MNIST, utils
from PIL import Image
import shutil
from utils.language_utils import word_to_indices, letter_to_vec

''' The femnist dataset is naturally non-iid.
    The ShakeSpeare dataset is naturally non-iid.
'''

class FEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): 数据集根目录
            train (bool): 是否加载训练集
            transform (callable, optional): 可选的图像变换
        """
        self.root = root
        self.train = train
        self.transform = transform

        # 确定数据文件夹路径
        self.data_dir = os.path.join(root, 'train' if train else 'test')

        # 存储所有样本
        self.samples = []
        self.labels = []

        # 加载数据
        self._load_data()
        self.targets = self.labels

    def _load_data(self):
        """
        从JSON文件中加载FEMNIST数据
        """
        for json_file in os.listdir(self.data_dir):
            if json_file.endswith('.json'):
                file_path = os.path.join(self.data_dir, json_file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # 遍历每个用户的数据
                    for user in data['users']:
                        user_samples = data['user_data'][user]['x']
                        user_labels = data['user_data'][user]['y']

                        # 将数据转换为numpy数组
                        for sample, label in zip(user_samples, user_labels):
                            # 28x28的灰度图像
                            sample_array = np.array(sample, dtype=np.uint8).reshape(28, 28)
                            # sample_tensor = torch.tensor(sample, dtype=torch.float32).reshape(1, 28, 28)

                            if self.transform:
                                sample_tensor = self.transform(sample_array)

                            self.samples.append(sample_tensor)
                            self.labels.append(label)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本

        Returns:
            tuple: (image, label)
        """
        return self.samples[idx], self.labels[idx]



class ShakeSpeare(Dataset):
    def __init__(self, data_root, train):
        super(ShakeSpeare, self).__init__()

        self.train_data_folder = os.path.join(data_root, "train")
        self.test_data_folder = os.path.join(data_root, "test")

        # train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/shakespeare/train",
        #                                                                          "./data/shakespeare/test")
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(self.train_data_folder, self.test_data_folder)
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


# if __name__ == '__main__':
#     # test = ShakeSpeare(train=True)
#     test = FEMNIST(train=True)
#     x = test.get_client_dic()
#     print("client_dic: " + str(len(x)))
#     for i in range(100):
#         print(len(x[i]))



# class FEMNIST(MNIST): # 一次性读取了全部数据
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
#         self.download = download
#         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
#         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'
#         self.train = train
#         self.root = root
#         self.training_file = f'{self.root}/femnist_train.pt'
#         self.test_file = f'{self.root}/femnist_test.pt'
#         self.user_list = f'{self.root}/femnist_user_keys.pt'

#         if not os.path.exists(f'{self.root}/femnist_test.pt') \
#                 or not os.path.exists(f'{self.root}/femnist_train.pt'):
#             if self.download:
#                 self.dataset_download()
#             else:
#                 raise RuntimeError('Dataset not found, set parameter download=True to download')

#         if self.train:
#             data_file = self.training_file
#         else:
#             data_file = self.test_file

#         data_targets_users = torch.load(data_file)
#         self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]
#         self.user_ids = torch.load(self.user_list)

#     def __getitem__(self, index):
#         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
#         img = Image.fromarray(img.numpy(), mode='F')
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target, user

#     def dataset_download(self):
#         paths = [f'{self.root}/raw/', f'{self.root}/processed/']
#         for path in paths:
#             if not os.path.exists(path):
#                 os.makedirs(path)

#         # download files
#         filename = self.download_link.split('/')[-1]
#         utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/raw/', filename=filename, md5=self.file_md5)

#         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
#         for file in files:
#             # move to processed dir
#             shutil.move(os.path.join(f'{self.root}/raw/', file), f'{self.root}/processed/')