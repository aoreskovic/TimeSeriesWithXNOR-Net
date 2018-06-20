import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class _BinActive(torch.autograd.Function):
    """ Binary activation function, using signum and replacing
        the backward gradinet with square approximation
    """

    def forward(self, input):
        self.save_for_backward(input)
        # In the forward pass use signum
        output = torch.sign(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        # In the backward pass using square
        # approximation of signum derivation
        grad_output[input.ge(1)] = 0.0
        grad_output[input.le(-1)] = 0.0
        return grad_output


class BinActive(nn.Module):
    def forward(self, x):
        return _BinActive()(x)


class BinConv2D(nn.Module):
    """ BinConv2D is XNOR-BitCount version
        of standard 2D convolution
    """

    def __init__(self, in_channels, out_channels, 
                kernel_size=3, stride=1, padding=0,
                bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.activ = BinActive()

    def forward(self, x):
        x1 = self.activ(x)
        x2 = self.conv(x1)
        x3 = self.activ(x2)
        return x3, x2


class BinLinear(nn.Module):
    """ BinLinear is XNOR-BitCount version
        of standard linear layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activ = BinActive()

    def forward(self, x):
        x1 = self.activ(x)
        x2 = self.linear(x1)
        x3 = self.activ(x2)
        return x3, x2


def DistanceFromPenalty(netParams, what):
    """ Function that introduces te additional cost toooptimize

    Arguments:
        netParams {Network Parameters} -- Parameters of the network
        what {int} -- calculate distance from which number
    """
    sum = 0
    i = 0
    for param in netParams:
        if (i % 2) == 0:
            distance = what-torch.abs(param)
            distance_squared = torch.pow(distance, 2)
            sum += torch.sum(distance_squared)
        i += 1
    return sum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """ Function that saves the nural net and optimizer state

    Arguments:
        state {dictionary} -- Current state of the neural net
        is_best {bool} -- Is the current iteration best one yer

    Keyword Arguments:
        filename {str} -- Name of the file (default: {'checkpoint.pth.tar'})
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'net_best.pth.tar')
