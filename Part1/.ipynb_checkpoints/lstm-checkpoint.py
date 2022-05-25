from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.param = dict({"seq_length": seq_length,
                           "input_dim": input_dim,
                           "hidden_dim": hidden_dim,
                           "output_dim": output_dim,
                           "batch_size": batch_size, })
        self.W_gx = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.randn(hidden_dim))

        self.W_ix = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.randn(hidden_dim))

        self.W_fx = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.randn(hidden_dim))

        self.W_ox = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.randn(hidden_dim))

        self.W_ph = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b_p = nn.Parameter(torch.randn(output_dim))

        self.h = None
        self.c = None

    def forward(self, x):
        # Implementation here ...
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h = torch.zeros(self.param["batch_size"], self.param["hidden_dim"]).to(device)
        self.c = torch.zeros(self.param["batch_size"], self.param["hidden_dim"]).to(device)
        for t in range(self.param["seq_length"]):
            g = torch.tanh(x[:, t:t+1] @ self.W_gx + self.h @ self.W_gh + self.b_g)
            i = torch.sigmoid(x[:, t:t+1] @ self.W_ix + self.h @ self.W_ih + self.b_i)
            f = torch.sigmoid(x[:, t:t + 1] @ self.W_fx + self.h @ self.W_fh + self.b_f)
            o = torch.sigmoid(x[:, t:t + 1] @ self.W_ox + self.h @ self.W_oh + self.b_o)
            self.c = torch.mul(g, i) + torch.mul(self.c, f)
            self.h = torch.mul(torch.tanh(self.c), o)
        return self.h @ self.W_ph + self.b_p
        
    # add more methods here if needed
