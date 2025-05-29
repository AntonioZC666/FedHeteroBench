import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, n_outputs * 2)
        # self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(n_outputs * 2, n_outputs * 2)
            for _ in range(1)])
        self.output = nn.Linear(n_outputs * 2, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        # x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            # x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x