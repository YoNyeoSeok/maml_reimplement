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

#class MAML():
#    def __init__(self, *args, **kwargs):
#        self.dataset = args[]
#        self.model = Construct_Sequential(args[])
#        self.meta_batch_size
#        self.meta_update_num
#
#
#    def meta_update(self):
#        pass
#
#    def train(self):
#        for i in range(self.meta_update_num):
#            for j in range(meta_batch_size):
#                if j == 0:
#                    
#        pass


dict = OrderedDict([
	('linear1', nn.Linear(1, 2)),
	('sigmoid1', nn.Sigmoid()),
	('linear2', nn.Linear(2, 1))])

dict = OrderedDict([
	('conv1', nn.Conv2d(1, 64, (3, 3))),
	('bn1', nn.BatchNorm2d(64)),
	('relu1', nn.ReLU()),
	('pool1', nn.MaxPool2d((2, 2))),
	('conv2', nn.Conv2d(64, 64, (3, 3))),
	('bn2', nn.BatchNorm2d(64)),
	('relu2', nn.ReLU()),
	('pool2', nn.MaxPool2d((2, 2))),
	('conv3', nn.Conv2d(64, 64, (3, 3))),
	('bn3', nn.BatchNorm2d(64)),
	('relu3', nn.ReLU()),
	('pool3', nn.MaxPool2d((2, 2)))
    ])


model = Construct_Sequential(dict)
#print('parameters', model._parameters)
print(model)
x = torch.rand((1, 1, 28, 28))
print(x.shape)
print(model(x).shape)

#for name, parameter in model.named_parameters():
#    print(name, parameter)
#    weights_dict[name] = parameter


