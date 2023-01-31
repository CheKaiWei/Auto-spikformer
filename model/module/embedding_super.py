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

    # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim
    def forward(self, x):
        T, B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = F.conv2d(x.flatten(0, 1), self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        
        if self.scale:
            return x * self.sampled_scale
        # _, H, W = x.shape
        # print(x.shape)
        x = x.reshape(T, B, -1, H//16, W//16).contiguous() # H, W needed change!
        x = self.lif(x) 
        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x

    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops