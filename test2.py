import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class Construct_Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(Construct_Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
                
    def forward(self, x, weights=None):
        if weights is None:
            return super(Construct_Sequential, self).forward(x)
        else:
            for name, module in self._modules.items():
                if isinstance(module, nn.Linear):
                    x = F.linear(x, weights[name + '.weight'], 
                                weights[name + '.bias'])
                elif isinstance(module, nn.ReLU):
                    x = F.relu(x)
                elif isinstance(module, nn.Sigmoid):
                    x = torch.sigmoid(x)
                else:
                    raise KeyError("Not Expedted Module '{}'".format(module))
            return x

dict = OrderedDict([
	('linear1', nn.Linear(1, 2)),
	('sigmoid1', nn.Sigmoid()),
	('linear2', nn.Linear(2, 1))])

model = Construct_Sequential(dict)
#print('parameters', model._parameters)

x = torch.rand(1)
#print(x)
print(model(x))
#print('parameters', model._parameters)

weights_dict = OrderedDict()
#for (key, module) in model._modules.items():
#    print(key, module)
for name, parameter in model.named_parameters():
    #    print(name, parameter)
    weights_dict[name] = parameter
#print(weights_dict)

print(model(x, weights_dict))


#print('parameters', model._parameters)
