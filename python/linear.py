import torch
import torch.nn as nn
from matmul_autograd import matmul


class MyLinear(nn.Module):

    def __init__(self,in_features,out_features):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_features,out_features)
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self,x):

        y = matmul(x,self.weight)

        return y + self.bias