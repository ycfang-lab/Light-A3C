from __future__ import print_function
import torch
import math
import torch.nn as nn
from torch.nn import functional as F


class BinActive(torch.autograd.Function):
    """
    Binarize the input activations and calculate the mean across channel dimension.
    """

    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Module):  # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            if self.previous_conv:
                self.bn = nn.BatchNorm2d(input_channels // size, eps=1e-4, momentum=0.1, affine=True)
            else:
                self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x


class BinNoisy(nn.Module):  # change the name of BinConv2d
    def __init__(self, input_channels, output_channels, std_init=0.4):
        super(BinNoisy, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.layer_type = 'BinNoisy'
        self.std_init = std_init
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels))
        self.weight_sigma = nn.Parameter(torch.empty(output_channels, input_channels))
        self.register_buffer('weight_epsilon', torch.empty(output_channels, input_channels))
        self.register_parameter('bias', None)
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            x = F.linear(x, self.weight + self.weight_sigma * self.weight_epsilon,
                         self.bias)
            return x
        else:
            x = F.linear(x, self.weight, self.bias)
            return x

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_channels)
        self.weight.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_channels))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_channels)
        epsilon_out = self._scale_noise(self.output_channels)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
