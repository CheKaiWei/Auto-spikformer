import torch
from torch import nn
from model.module.BN_super import BNSuper
class MLP(nn.Module):
    def __init__(self, out_features=6, drop=0.):
        super().__init__()
        self.sampled_bn_weight = None
        self.sample_bn_bias = None
        self.sample_bn_mean = None
        self.sampled_bn_std = None

        self.fc1_bn = nn.BatchNorm2d(out_features)
        torch.nn.init.uniform_(self.fc1_bn.weight)
        torch.nn.init.uniform_(self.fc1_bn.bias)

    def set_sample_config(self, sample_embed_dim=4):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_bn_weight = self.fc1_bn.weight[:self.sample_embed_dim, ...] ## TODO whether I can use this function to do n1?
        self.sampled_bn_bias = self.fc1_bn.bias[:self.sample_embed_dim, ...]
        self.sampled_bn_mean = self.fc1_bn.running_mean[:self.sample_embed_dim, ...]
        self.sampled_bn_std = self.fc1_bn.running_var[:self.sample_embed_dim, ...]


    
    # def set_bn_sample_
    def forward(self, x):

        x_original = self.fc1_bn(x)
        print('original result: ',x_original)

        x_f = nn.functional.batch_norm(x[:,:4,...], running_mean=self.sampled_bn_mean, running_var=self.sampled_bn_std,\
             weight=self.sampled_bn_weight, bias=self.sampled_bn_bias, training=True, momentum=0.9)
        print('functional result: ',x_f)

        x_f = nn.functional.batch_norm(x[:,:4,...], running_mean=self.sampled_bn_mean, running_var=self.sampled_bn_std, training=True, momentum=0.9)
        print('functional result2: ',x_f)
        return x_original

    def show(self):
        print('bn:',self.fc1_bn)
        print(self.fc1_bn.weight.grad)
        print(self.fc1_bn.bias.grad)
        print(self.fc1_bn.running_mean)
        print(self.fc1_bn.running_var)

mlp = MLP()
mlp.set_sample_config()
mlp.show()
input_tensor = torch.randn((2,6,5,5)).requires_grad_(True)
x_original = mlp(input_tensor)
loss = torch.mean(x_original*100)
loss.backward()
mlp.show()


print('----------------')
bn_super = BNSuper(super_out_dim=6)
bn_super.show()
input_tensor = torch.randn((2,4,5,5)).requires_grad_(True)
bn_super.set_sample_config(4)
x_original = bn_super(input_tensor)
loss = torch.mean(x_original*100)
loss.backward()
bn_super.show()
