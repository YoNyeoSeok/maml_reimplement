import torch
from torch.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, weight=None, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        
