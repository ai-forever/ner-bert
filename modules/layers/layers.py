import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        I.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)
