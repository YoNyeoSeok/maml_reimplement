import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torchviz import make_dot

class Change_view(nn.Module):
    def __init__(self, size):
        super(Change_view, self).__init__()
        self.size = size
    def forward(self, x, size=None):
        if size is None:
            size = self.size
        x = x.view(size)
        return x

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
                    if module.bias is None:
                        x = F.linear(x, weights[name + '.weight'], module.bias)
                    else:
                        x = F.linear(x, weights[name + '.weight'], 
                                weights[name + '.bias'])

                elif isinstance(module, nn.ReLU):
                    x = F.relu(x)
                elif isinstance(module, nn.Sigmoid):
                    x = torch.sigmoid(x)
                elif isinstance(module, nn.Conv2d):
                    x = F.conv2d(x, weights[name + '.weight'],
                                weights[name + '.bias'])
                elif isinstance(module, nn.BatchNorm2d):
                    #print(self._modules[name].__dict__)
                    x = F.batch_norm(x, None, None, weight=weights[name+'.weight'],
                            bias=weights[name+'.bias'], training=True)
                elif isinstance(module, nn.MaxPool2d):
                    #print(self._modules[name].__dict__)
                    x = self._modules[name](x)
                elif isinstance(module, Change_view):
                    #print(self._modules[name])
                    x = module(x)
                else:
                    raise KeyError("Not Expedted Module '{}'".format(module))
            return x

    def _copy(self, weights):
        for name, param in self.named_parameters():
            param.data = weights[name].data

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

dict_ = OrderedDict([
    ('linear1', nn.Linear(1, 2, bias=None)),
    ('sigmoid1', nn.Sigmoid()),
    ('linear2', nn.Linear(2, 1, bias=None))
    ])
model = Construct_Sequential(dict_)
fast_net = Construct_Sequential(dict_)
w0 = {}
for name, module in model._modules.items():
    if isinstance(module, nn.Linear):
        torch.nn.init.ones_(module.weight)
for name, param in model.named_parameters():
    w0[name] = param
    print(name, param.data)


alpha = .1
beta = .1
optimizer = torch.optim.SGD(model.parameters(), beta)

fast_net._copy(w0)
x = torch.tensor([[2.], [3.]]).requires_grad_(True)
x_ = torch.tensor([[2.1], [3.1]]).requires_grad_(True)
y = torch.tensor([0.])

print('predict')
predict = model(x)
print(predict)
dot = make_dot(predict, params=dict(list(model.named_parameters()) + [('x', x)]))
dot.render('first_predict')

print('loss')
loss = (predict-y)**2
print(loss)
dot = make_dot(loss, params=dict(list(model.named_parameters()) + [('x', x)]))
dot.render('first_loss')

meta_loss_ = []
for i in range(len(x)):
    grads = torch.autograd.grad(loss[i], fast_net.parameters(), create_graph=True)#, retain_graph=True)
    w_ = OrderedDict([ (name, w0[name] - alpha*grad)
        for name, grad in zip(w0, grads)
        ])
#    print('grad compute: alpha*grads')
#    for grad in grads:
#        print(grad*alpha)
#    print([grad.reshape(-1) for grad in grads])
#    print(torch.cat([grad.reshape(-1) for grad in grads]))
#    print(torch.sum(torch.cat([grad.reshape(-1) for grad in grads])))
    dot = make_dot(torch.sum(torch.cat([grad.reshape(-1) for grad in grads])), params=dict(list(fast_net.named_parameters()) + [('x', x)]))
    dot.render('grads')
    
#    print('compare')
#    for name, param in fast_net.named_parameters():
#        print(name, param.data, w_[name])
   
    print('meta predict')
    meta_predict = fast_net.forward(x_[i], w_)
    print(meta_predict)
#    dot = make_dot(meta_predict, params=dict(list(fast_net.named_parameters()) + [('x', x)]))
#    dot.render('meta_predict')

    meta_loss_.append((meta_predict-y)**2)

meta_loss = torch.mean(torch.cat(meta_loss_))
print('meta_loss')
print(meta_loss)
dot = make_dot(meta_loss, params=dict(list(fast_net.named_parameters()) + [('x', x), ('x_', x_)]))
dot.render('meta_loss')


print('meta_loss grad')
if True:
    optimizer.zero_grad()
    meta_loss.backward()
    for name, param in model.named_parameters():
        print(name, param.grad)
    optimizer.step()
else:
    grads = torch.autograd.grad(meta_loss, model.parameters())#, retain_graph=True)
    w_ = OrderedDict([ (name, w0[name] - alpha*grad)
        for name, grad in zip(w0, grads)
        ])
    model._copy(w_)
    print(grads)

print('model')
for name, param in model.named_parameters():
    print(name, param.data, param.grad)

result_predict = model(x)
print('result_predict')
print(model(x))
#print(model(x))

dot = make_dot(result_predict, params=dict(list(model.named_parameters()) + [('x', x)]))
dot.render('result_predict')

"""

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
	('pool3', nn.MaxPool2d((2, 2))),
    ])
dict2 = OrderedDict([
    ('flatten1', Change_view((2, 64)))
    ])
dict3 = OrderedDict([
    ('fc1', nn.Linear(64, 5))
    ])

print(dict2.update(dict3))
print(dict.update(dict2))

model = Construct_Sequential(dict)
feature = Construct_Sequential(dict)
#print('parameters', model._parameters)
print(model)
#x = torch.rand((1, 1, 28, 28))
x = torch.rand((2, 1, 28, 28))
print(x.shape)
y = model(x)
print(y.shape)


#for name, parameter in model.named_parameters():
#    print(name, parameter)
#    weights_dict[name] = parameter

"""
