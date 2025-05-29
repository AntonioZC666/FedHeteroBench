import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fedimpro.GaussianSynthesisLabel import GaussianSynthesisLabel

class FeatureGenerator(nn.Module):
    def __init__(self, args):
        super(FeatureGenerator, self).__init__()
        self.args = args
        self.device = self.args.device
        self.num_classes = self.args.num_classes


        if self.args.dataset == 'cifar10':
            self.predefined_number_per_class = 5000
        elif self.args.dataset == 'fmnist':
            self.predefined_number_per_class = 6000
        elif self.args.dataset == 'svhn':
            self.predefined_number_per_class = 7000
        elif self.args.dataset == 'cifar100':
            self.predefined_number_per_class = 600
        elif self.args.dataset == 'femnist':
            self.predefined_number_per_class = 10000
        else:
            raise NotImplementedError


        self.forward_count = 0

        self.feature_synthesis = GaussianSynthesisLabel(self.args)

    def sample(self, x=None, labels=None):
        align_features, align_labels = self.feature_synthesis.sample(x, labels)
        return align_features, align_labels

    def initial_model_params(self, feat, feat_length=None):
        self.feature_synthesis.initial_model_params(feat, feat_length)

    def update(self, feat=None, labels=None):
        decode_error = 0.0
        decode_error = self.feature_synthesis.update(feat, labels)

        return decode_error


