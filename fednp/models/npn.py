import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NPNLinearLite(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels, dual_input = True, init_type = 0):
        super(NPNLinearLite, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_input = dual_input

        self.W_m = torch.nn.Parameter(2 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * (torch.rand(in_channels, out_channels) - 0.5)) # ???
        self.bias_m = torch.nn.Parameter(torch.zeros(out_channels))


    def forward(self, x):
        device = self.W_m.device  # added
        if self.dual_input:
            x_m, x_s = x
        else:
            x_m = x
            x_s = x.clone()
            x_s = 0 * x_s

        x_m = x_m.to(device) # added
        x_s = x_s.to(device) # added

        o_m = torch.mm(x_m.unsqueeze(0), self.W_m)
        o_m = o_m + self.bias_m.expand_as(o_m)

        o_s = torch.mm(x_s.unsqueeze(0), self.W_m * self.W_m)

        # o_m, o_s = o_s.to(self.args.device), o_m.to(self.args.device)

        # print(self.W_m.device)
        # print(o_m.device, o_s.device, x_m.device, x_s.device)

        return o_m.squeeze(0), o_s.squeeze(0)


class NPNSigmoid(torch.nn.Module):
    def __init__(self):
        super(NPNSigmoid, self).__init__()
        self.xi_sq = math.pi / 8
        self.alpha = 4 - 2 * math.sqrt(2)
        self.beta = - math.log(math.sqrt(2) + 1)

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = torch.sigmoid(o_m / (1 + self.xi_sq * o_s) ** 0.5)
        a_s = torch.sigmoid(self.alpha * (o_m + self.beta) / (1 + self.xi_sq * self.alpha ** 2 * o_s) ** 0.5) - a_m ** 2
        return a_m, a_s


class NPN(torch.nn.Module):
    def __init__(self, args, model):
        super(NPN, self).__init__()
        self.model = model
        dim = self.model.linear.weight.numel()
        self.net = torch.nn.Sequential(
            NPNLinearLite(args, 10, 10, True),
            NPNSigmoid(),
            NPNLinearLite(args, 10, dim, True)
        )
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    model = Hybrid(dim = 100 * 512).cuda()
    mu = torch.ones(1).cuda()
    sigma = torch.zeros(1).cuda()

    mu, sigma = model((mu, sigma))
    print(mu.shape, sigma.shape)
