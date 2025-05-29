import math
import torch
from torch import nn
from torch.utils.data import DataLoader

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)

class Client:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        self.data_length = len(self.data_index) # length of dataset

    # for training
    def train(self, net, npn_model, cavity):

        net.train()
        npn_model.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        optimizer_npn = torch.optim.SGD(npn_model.parameters(),
                                        lr=lr,
                                        weight_decay=self.args.weight_decay)

        cavity_mu, cavity_sigma = cavity
        # cavity_mu, cavity_sigma = cavity_mu.to(self.args.device), cavity_sigma.to(self.args.device)

        with torch.no_grad():
            fx_all = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, fx = net(images)  # TODO 模型修改
                fx_all.append(fx)
            fx_all = torch.cat(fx_all).to(self.args.device)
            mu, sigma = self.get_ep_prior(cavity_mu, cavity_sigma, fx_all)
            # print("mu: {} , sigma: {}".format(mu, sigma))

        for iter in range(local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs, fx = net(images)

                kld_loss = torch.mean(-0.5 * (1 + torch.log(torch.sqrt(sigma + EPS).mean()) - mu.mean() ** 2 - torch.sqrt(sigma + EPS).mean()))
                # print(kld_loss)
                # print("-------------")

                mu, sigma = npn_model((mu, sigma))
                # print("after NPN: mu: {} , sigma: {}".format(mu, sigma))
                # print("-------------")
                mu, sigma = mu.to(self.args.device), sigma.to(self.args.device)
                loss_np = ((net.linear.weight.reshape(-1) - mu) ** 2 / (2 * sigma + EPS)).mean()  # TODO , 涉及到模型

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                optimizer_npn.zero_grad()

                (loss +  self.args.mu * (loss_np + kld_loss)).backward()

                optimizer.step()
                optimizer_npn.step()

                mu, sigma = self.get_ep_prior(cavity_mu, cavity_sigma, fx.detach())
                # print("mu: {} , sigma: {}".format(mu, sigma))
                # print("-------------")


        with torch.no_grad():
            fx_all = []
            for i, (X, y) in enumerate(self.train_loader):
                X = X.cuda()
                _, fx = net(X)
                fx_all.append(fx)
            mu, sigma = self.get_ep_prior(cavity_mu, cavity_sigma, torch.cat(fx_all))
            mu, sigma = self.remove_cavity(mu, sigma, cavity_mu, cavity_sigma)

        return net.state_dict(), mu, sigma, self.data_length



    def calc_template(self, a, b, c, d):
        return (((a + b) * c) - (b * d))


    def calc_statistics(self, x, y, a, b):
        ksei_square = math.pi / 8
        nu = a * (x + b)
        de = torch.sqrt(torch.relu(1 + ksei_square * a * a * y))
        return torch.sigmoid(nu / (de + EPS))


    def get_ep_prior(self, theta_m, theta_s, fx):
        fx = fx.mean().squeeze()
        fxsq = fx * fx
        cm, cs = fx * theta_m, fxsq * theta_s

        e_1 = self.calc_statistics(cm, cs, 1, 0)
        e_2 = self.calc_statistics(cm, cs, 4 - 2 * SQRT2, math.log(SQRT2 + 1))
        e_3 = self.calc_statistics(cm, cs, 6 * (1 - 1 / POW2_3O2), math.log(POW2_3O2 - 1))

        _p_1 = self.calc_template(cm, cs, e_1, e_2)
        _p_2 = self.calc_template(cm, 2 * cs, e_2, e_3) ###
        s_0 = e_1
        s_1 = _p_1 / (s_0 * fx + EPS)
        s_2 = (cs * e_1 + self.calc_template(cm, cs, _p_1, _p_2)) / (s_0 * fxsq + EPS)

        theta_m, theta_s = s_1, torch.relu(s_2 - s_1 * s_1) + EPS
        theta_m = torch.clamp(theta_m, EPS, 5)
        theta_s = torch.clamp(theta_s, EPS, 5)
        del cm, cs, e_1, e_2, e_3, s_0, s_1, s_2
        return theta_m, theta_s


    def remove_cavity(self, tm, ts, cm, cs):
        tb = tm / (ts + EPS)
        td = -0.5 / (ts + EPS)
        cb = cm / (cs + EPS)
        cd = -0.5 / (cs + EPS)
        qb = tb - cb
        qd = torch.relu(td - cd) + EPS
        qs = - 2  / (qd + EPS)
        qm = qb * qs
        qm = torch.clamp(qm, EPS, 5)
        qs = torch.clamp(qs, EPS, 5)
        return qm, qs