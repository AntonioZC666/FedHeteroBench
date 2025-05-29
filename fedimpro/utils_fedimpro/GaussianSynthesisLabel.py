import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GaussianSynthesisLabel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = self.args.device
        self.num_classes = self.args.num_classes

        self.predefined_number_per_class = 5000
        self.forward_count = 0

        self.feature_means = None
        self.feature_std = None

    def sample(self, x=None, labels=None):
        repeat_times = x.size(0) // self.num_classes + 1
        fake_labels = torch.tensor(list(range(0, self.num_classes))*repeat_times).to(self.device)
        fake_features = self.feature_means.repeat(repeat_times, 1).to(self.device)
        fake_std = self.feature_std.repeat(repeat_times, 1).to(self.device)
        fake_features = torch.normal(mean=fake_features, std=fake_std)

        return fake_features, fake_labels


    def initial_model_params(self, feat, feat_length=None):
        self.feature_means = torch.rand(self.num_classes, feat.shape[1])
        self.feature_std = torch.ones([self.num_classes, feat.shape[1]]) * 0.1
        self.feature_means = self.feature_means.to(self.device)
        self.feature_std = self.feature_std.to(self.device)

    def update(self, feat=None, labels=None):
        dif_error = 0.0
        with torch.no_grad():
            for label in range(self.num_classes):
                if torch.any(labels==label):

                    feat_mean_label = feat[labels==label].mean(dim=0)
                    feat_std_label = feat[labels==label].std(dim=0)
                    dif_error += (feat_mean_label - self.feature_means[label]).norm(p=2)
                    weight = min(1.0, feat_mean_label.size(0)/self.predefined_number_per_class)

                    self.feature_means[label] = self.feature_means[label]*(1 - weight) \
                        + feat_mean_label*weight
                    if not torch.any(torch.isnan(feat_std_label)):
                        self.feature_std[label] = self.feature_std[label]*(1 - weight) \
                            + feat_std_label*weight
                else:
                    pass

                self.feature_std[self.feature_std > 0.1] = 0.1

        return dif_error.item()