import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BNSuper(nn.Module):
    def __init__(self, super_out_dim, dim_1d_2d='2d', bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__() ## TODO check!

        self.super_out_dim = super_out_dim
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale

        if dim_1d_2d == '1d':
            self.bn = nn.BatchNorm1d(super_out_dim)
        else:
            self.bn = nn.BatchNorm2d(super_out_dim)

        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode


    def set_sample_config(self, sample_out_dim):
        self.sample_out_dim = sample_out_dim

        self.samples['weight'] = self.bn.weight[:self.sample_out_dim, ...] ## TODO whether I can use this function to do n1?
        self.samples['bias'] = self.bn.bias[:self.sample_out_dim, ...]
        self.samples['mean'] = self.bn.running_mean[:self.sample_out_dim, ...]
        self.samples['var'] = self.bn.running_var[:self.sample_out_dim, ...]

        return self.samples

    def forward(self, x, training=True):
        return nn.functional.batch_norm(x, running_mean=self.samples['mean'], running_var=self.samples['var'],\
             weight=self.samples['weight'], bias=self.samples['bias'], training=training, momentum=0.9)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops

    def show(self):
        print('bn:',self.bn)
        print(self.bn.weight.grad)