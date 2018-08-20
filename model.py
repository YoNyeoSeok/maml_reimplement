import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple(nn.Module):
    def __init__(self, d_input=1, d_hidden=2, d_output=1):
        super(Simple, self).__init__()
        
        self.hidden = nn.Linear(d_input, d_hidden)
        self.predict = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


    def copy_(self, src):
        for param, param_ in zip(self.parameters(), src.parameters()):
            param.requires_grad = False
            param.copy_(param_)
            param.retain_grad()

        return self



    #    class MAML(nn.module):

