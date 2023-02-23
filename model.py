import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
from BN_super import BNSuper
from Linear_super import LinearSuper
# from sj_dropout import SN_Droppath
__all__ = ['spikformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1_linear = nn.Linear(in_features, hidden_features)
        # self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_linear = LinearSuper(super_in_dim=in_features, super_out_dim=hidden_features)
        self.fc1_bn = BNSuper(super_out_dim=hidden_features, dim_1d_2d='1d')
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.fc2_linear = nn.Linear(hidden_features, out_features)
        # self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_linear = LinearSuper(super_in_dim=hidden_features, super_out_dim=out_features)
        self.fc2_bn = BNSuper(super_out_dim=out_features, dim_1d_2d='1d')
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        self.c_hidden = x.shape[-1]
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.super_embed_dim = dim
        # self.q_linear = nn.Linear(dim, dim)
        # self.q_bn = nn.BatchNorm1d(dim)
        self.q_linear = LinearSuper(self.super_embed_dim, self.super_embed_dim)
        self.q_bn = BNSuper(super_out_dim=self.super_embed_dim, dim_1d_2d='1d')
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.k_linear = nn.Linear(dim, dim)
        # self.k_bn = nn.BatchNorm1d(dim)
        self.k_linear = LinearSuper(self.super_embed_dim, self.super_embed_dim)
        self.k_bn = BNSuper(super_out_dim=self.super_embed_dim, dim_1d_2d='1d')
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.v_linear = nn.Linear(dim, dim)
        # self.v_bn = nn.BatchNorm1d(dim)
        self.v_linear = LinearSuper(self.super_embed_dim, self.super_embed_dim)
        self.v_bn = BNSuper(super_out_dim=self.super_embed_dim, dim_1d_2d='1d')
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        # self.proj_linear = nn.Linear(dim, dim)
        # self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_linear = LinearSuper(self.super_embed_dim, self.super_embed_dim)
        self.proj_bn = BNSuper(super_out_dim=self.super_embed_dim, dim_1d_2d='1d')
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None):
        # print('setting !!!!!!!!!!!!!!!')
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads

        self.sample_qk_embed_dim = self.sample_in_embed_dim

        # if not self.change_qkv:
        #     # self.sample_qk_embed_dim = self.super_embed_dim
        #     self.sample_qk_embed_dim = self.sample_in_embed_dim
        #     self.sample_scale = (self.sample_in_embed_dim // self.sample_num_heads) ** -0.5

        # else:
        #     self.sample_qk_embed_dim = sample_q_embed_dim
        #     self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5
        # print(sample_in_embed_dim, 3*self.sample_qk_embed_dim)
        # self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3*self.sample_qk_embed_dim)
        
        self.q_linear.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=self.sample_qk_embed_dim)
        self.k_linear.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=self.sample_qk_embed_dim)
        self.v_linear.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=self.sample_qk_embed_dim)
        self.proj_linear.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)

        # if self.relative_position:
        #     self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
        #     self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)

        self.q_bn.set_sample_config(self.sample_qk_embed_dim)
        self.k_bn.set_sample_config(self.sample_qk_embed_dim)
        self.v_bn.set_sample_config(self.sample_qk_embed_dim)
        self.proj_bn.set_sample_config(self.sample_in_embed_dim)

    def forward(self, x):
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, \
                          sample_attn_dropout=None, sample_out_dim=None, sample_threshold=None, sample_tau=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        # self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)

        self.mlp.fc1_linear.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.mlp.fc2_linear.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)
        self.mlp.fc1_bn.set_sample_config(self.sample_ffn_embed_dim_this_layer)
        self.mlp.fc2_bn.set_sample_config(self.sample_out_dim)
        
        self.mlp.fc1_lif.threshold = sample_threshold
        self.mlp.fc1_lif.tau = sample_tau
        self.mlp.fc2_lif.tau = sample_tau
        self.attn.q_lif.tau = sample_tau
        self.attn.k_lif.tau = sample_tau
        self.attn.v_lif.tau = sample_tau
        self.attn.attn_lif.tau = sample_tau
        self.attn.proj_lif.tau = sample_tau

        # self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x):
        if self.is_identity_layer:
            return x
        x_attn = self.attn(x)
        x = x + x_attn
        x = x + self.mlp(x)

        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_bn3 = BNSuper(super_out_dim=embed_dims, dim_1d_2d='2d')
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_bn = BNSuper(super_out_dim=embed_dims, dim_1d_2d='2d')
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj_conv3.weight[:self.sample_embed_dim, ...]
        self.sampled_rep_weight = self.rpe_conv.weight[:self.sample_embed_dim, :self.sample_embed_dim, ...]
        # print('!!',self.sample_embed_dim)
        self.proj_bn3.set_sample_config(self.sample_embed_dim)
        self.rpe_bn.set_sample_config(self.sample_embed_dim)


    def forward(self, x):
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

        # print('input3',x.shape)
        # x = self.proj_conv3(x)
        x = F.conv2d(x, self.sampled_weight, stride=self.proj_conv3.stride, padding=self.proj_conv3.padding)
        # print('output3',x.shape)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        # x = self.rpe_conv(x)
        x = F.conv2d(x, self.sampled_rep_weight, stride=self.rpe_conv.stride, padding=self.rpe_conv.padding)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        # self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.head = LinearSuper(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_sample_config(self, config: dict):
        # print(config)
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.T = config['time_step']
        self.sample_threshold = config['threshold']
        self.sample_tau = config['tau']

        patch_embed.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blk in enumerate(block):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                # sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                # sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                blk.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=0,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=0,
                                        sample_threshold=self.sample_threshold[i],
                                        sample_tau=self.sample_tau[i])
            # exceeds sample layer number
            else:
                blk.set_sample_config(is_identity_layer=True)
        # if self.pre_norm:
        #     self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

        # self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        # self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        # self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        # for i, blocks in enumerate(self.blocks):
        #     # not exceed sample layer number
        #     if i < self.sample_layer_num:
        #         sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
        #         sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
        #         blocks.set_sample_config(is_identity_layer=False,
        #                                 sample_embed_dim=self.sample_embed_dim[i],
        #                                 sample_mlp_ratio=self.sample_mlp_ratio[i],
        #                                 sample_num_heads=self.sample_num_heads[i],
        #                                 sample_dropout=sample_dropout,
        #                                 sample_out_dim=self.sample_output_dim[i],
        #                                 sample_attn_dropout=sample_attn_dropout)
        #     # exceeds sample layer number
        #     else:
        #         blocks.set_sample_config(is_identity_layer=True)
        # if self.pre_norm:
        #     self.norm.set_sample_config(self.sample_embed_dim[-1])
        # self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(2)

    def forward(self, x):
        # print('time step:',self.T)
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        # print(x.shape)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


