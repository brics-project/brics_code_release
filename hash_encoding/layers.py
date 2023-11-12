"""Layers to generate the hash table."""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.networks_stylegan2 import FullyConnectedLayer

class ModulatedConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, s_dim: int,
                 activation: nn.Module=None,
                 bias: bool=True, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.multiplier = 1 / np.sqrt(in_ch)
        weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size))
        self.register_parameter('weight', weight)
        nn.init.xavier_normal_(self.weight)

        if bias:
            bias = nn.Parameter(torch.zeros(out_ch))
            self.register_parameter('bias', bias)
        else:
            self.bias = None
        
        if activation is not None:
            self.activ = activation()
        else:
            self.activ = None

        self.s_mapping = FullyConnectedLayer(s_dim, in_ch, bias_init=1)

    def forward(self, x, s):
        """
            x: B x (N) x IN
            s: B x s_dim

        """
        batch_size = x.size(0)
        s = self.s_mapping(s)
        # NOTE: The batch size may be different
        s_batch_size = s.size(0)
        if s_batch_size < batch_size:
            s = s.repeat(batch_size // s_batch_size, 1)

        weight = self.weight
        w = weight.unsqueeze(dim=0) # 1 x OUT x IN x K
        w = w * s.reshape(batch_size, 1, -1, 1)
        decoefs = (w.square().sum(dim=[2, 3]) + 1e-8).rsqrt() # B x O

        s = s.unsqueeze(dim=2) if x.dim() == 3 else s
        decoefs = decoefs.unsqueeze(dim=2) if x.dim() == 3 else decoefs

        x = x * s
        x = F.conv1d(x, weight, padding=self.kernel_size//2, bias=self.bias) # B x (N) x O
        x = x * decoefs
        x = x * self.multiplier
        if self.activ is not None:
            x = self.activ(x)

        return x


class ModulatedLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s_dim: int,
                 activation: nn.Module=None,
                 bias: bool=True, **kwargs):
        super().__init__()
        weight = nn.Parameter(torch.randn(out_ch, in_ch))
        self.register_parameter('weight', weight)
        nn.init.xavier_normal_(self.weight)
        self.in_ch = in_ch

        if bias:
            bias = nn.Parameter(torch.zeros(out_ch))
            self.register_parameter('bias', bias)
        else:
            self.bias = None
        
        if activation is not None:
            self.activ = activation()
        else:
            self.activ = None

        self.s_mapping = FullyConnectedLayer(s_dim, in_ch, bias_init=1)

    def forward(self, x, s):
        """
            x: B x (N) x IN
            s: B x s_dim

        """
        batch_size = x.size(0)
        s = self.s_mapping(s)
        # NOTE: The batch size may be different
        s_batch_size = s.size(0)
        if s_batch_size < batch_size:
            s = s.repeat(batch_size // s_batch_size, 1)

        weight = self.weight
        # Pre-normalize
        weight = weight * (1 / np.sqrt(self.in_ch) / weight.norm(float('inf'), dim=[1,], keepdim=True)) # max_I
        s = s / s.norm(float('inf'), dim=1, keepdim=True) # max_I

        w = weight.unsqueeze(dim=0) # 1 x OUT x IN
        w = w * s.reshape(batch_size, 1, -1)
        decoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt() # B x O

        s = s.unsqueeze(dim=1) if x.dim() == 3 else s
        decoefs = decoefs.unsqueeze(dim=1) if x.dim() == 3 else decoefs

        x = x * s
        x = F.linear(x, weight, bias=self.bias) # B x (N) x O
        x = x * decoefs
        if self.activ is not None:
            x = self.activ(x)
        x = x.clamp(-255, 255)

        return x
