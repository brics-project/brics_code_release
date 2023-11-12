"""
Modules for generating hash tables
"""
from functools import partial

import torch.nn as nn
from hash_encoding.layers import ModulatedLinear


class StackedModulatedMLP(nn.Module):
    def __init__(self, in_ch: int, h_ch: int, out_ch: int, s_dim: int,
                 n_layers: int,
                 table_num: int=16,
                 in_activ: nn.Module=nn.ReLU,
                 out_activ: nn.Module=nn.Tanh,
                 use_layer_norm: bool=False):
        """
            Args:
                in_ch: input dimension
                h_ch: hidden dimension
                out_ch: output dimension
                s_dim: style code dimension
                n_layers: how many layers of MLPs in total
                    (including input and output layers)
                table_num (int): number of tables
                in_activ : inside (hidden layers) activation
                out_activ : output activation
                norm_layer (nn.Module): if Other normalization is used
                use_layer_norm (nn.Module): Use layer normalization
        """
        super().__init__()

        self.module_list = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        self.n_layers = n_layers
        linear_layer = partial(ModulatedLinear, table_num=table_num)
        for i in range(n_layers):
            if i == 0:
                self.module_list.append(linear_layer(in_ch, h_ch, s_dim, activation=in_activ))
            elif i == n_layers - 1:
                self.module_list.append(linear_layer(h_ch, out_ch, s_dim, activation=out_activ))
            else:
                self.module_list.append(linear_layer(h_ch, h_ch, s_dim, activation=in_activ))

    def forward(self, x, s):
        for i, m in enumerate(self.module_list):
            x = m(x, s)
        return x
