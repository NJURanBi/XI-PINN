# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 下午3:54
# @Author  : NJU_RanBi
import torch
import torch.nn as nn

class Vanilla_Net(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, depth):
        super(Vanilla_Net, self).__init__()
        self.depth = depth - 1
        self.list = nn.ModuleList()
        self.ln1 = nn.Linear(in_dim, h_dim)
        self.act = nn.Tanh()

        for i in range(self.depth):
            self.list.append(nn.Linear(h_dim, h_dim))

        self.lnd = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act(out)
        for i in range(self.depth):
            out = self.list[i](out)
            out = self.act(out)
        out = self.lnd(out)
        return out