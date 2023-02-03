import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import LIFNode

# LIFNode(tau=5., decay_input=False, v_threshold=0.5, detach_reset=True, surrogate_function=DSpike())


class PatchembedSuper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale
        self.lif = LIFNode(tau=2.0, detach_reset=True)

        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None


        self.sampled_bn_weight = None
        self.sample_bn_bias = None
        self.sample_bn_mean = None
        self.sample_bn_var = None

        self.proj_conv = nn.Conv2d(in_chans, embed_dim//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim//8)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True)

        self.proj_conv1 = nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dim//4)
        self.proj_lif1 = LIFNode(tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dim//2)
        self.proj_lif2 = LIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dim)
        self.proj_lif3 = LIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = LIFNode(tau=2.0, detach_reset=True)


    def set_sample_config(self, sample_embed_dim):
        # print('!!!!!',sample_embed_dim)
        # self.sample_embed_dim = sample_embed_dim
        # self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        # self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        # if self.scale:
        #     self.sampled_scale = self.super_embed_dim / sample_embed_dim

        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj_conv3.weight[:self.sample_embed_dim, ...]

        self.sampled_bn_weight = self.proj_bn3.weight[:self.sample_embed_dim, ...]
        self.sampled_bn_bias = self.proj_bn3.bias[:self.sample_embed_dim, ...]
        self.sampled_bn_mean = self.proj_bn3.running_mean[:self.sample_embed_dim, ...]
        self.sampled_bn_std = self.proj_bn3.running_var[:self.sample_embed_dim, ...]



        

    def forward(self, x):
        # T, B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # x = F.conv2d(x.flatten(0, 1), self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        
        # if self.scale:
        #     return x * self.sampled_scale
        # # _, H, W = x.shape
        # # print(x.shape)
        # x = x.reshape(T, B, -1, H//16, W//16).contiguous() # H, W needed change!
        # x = self.lif(x) 
        # x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        T, B, C, H, W = x.shape


        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        # x = self.proj_conv3(x)
        # x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        # x = self.proj_lif3(x).flatten(0, 1).contiguous()
        # x = self.maxpool3(x)

        x = F.conv2d(x, self.sampled_weight, stride=self.proj_conv3.stride, padding=self.proj_conv3.padding)
        x = nn.functional.batch_norm(x, running_mean=self.sampled_bn_mean, running_var=self.sampled_bn_std,\
             weight=self.sampled_bn_weight, bias=self.sampled_bn_bias, training=True, momentum=0.9)
        x = x.reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x

    def calc_sampled_param_num(self):
        # return  self.sampled_weight.numel() + self.sampled_bias.numel()
        return  self.sampled_weight.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops