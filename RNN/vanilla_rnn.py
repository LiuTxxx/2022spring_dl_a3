from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.param = dict({"seq_length": seq_length,
                           "input_dim": input_dim,
                           "hidden_dim": hidden_dim,
                           "output_dim": output_dim,
                           "batch_size": batch_size, })
        self.W_hx = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_ph = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b_h = nn.Parameter(torch.randn(hidden_dim))
        self.b_o = nn.Parameter(torch.randn(output_dim))
        self.h = torch.zeros(batch_size, hidden_dim)

    def forward(self, x):
        # Implementation here ...
        device = torch.device('cuda')
        self.h = torch.zeros(self.param["batch_size"], self.param["hidden_dim"]).to(device)
        for t in range(self.param["seq_length"]):
            self.h = torch.tanh(x[:, t:t+1] @ self.W_hx + self.h @ self.W_hh + self.b_h)
        return self.h @ self.W_ph + self.b_o

    # add more methods here if needed
