import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


import shutil

class _BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_output[input.ge(1)] = 0.0
        grad_output[input.le(-1)] = 0.0
        return grad_output


class BinActive(nn.Module):
    def forward(self, x):
        return _BinActive()(x)


class BinConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.activ = BinActive()

    def forward(self, x):
        #print("----------------------------------")
        #print(x)
        x1 = self.activ(x)
        #print(x1)
        x2 = self.conv(x1)
        #print(x2)
        x3 = self.activ(x2)
        #print(x3)
        return x3, x2

class BinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activ = BinActive()

    def forward(self, x):
        x1 = self.activ(x)
        x2 = self.linear(x1)
        x3 = self.activ(x2)
        return x3, x2

def DistanceFromPenalty(netParams, what):
    sum = 0
    for param in netParams:
        sum += torch.pow(torch.sum(what-torch.abs(param)),2)
        #sum += torch.abs(torch.sum(what-torch.abs(param)))
    return sum



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'net_best.pth.tar')