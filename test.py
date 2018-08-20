import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model import Simple
import copy
from torchviz import make_dot

alpha = .5



class MINI(nn.Module):
    def __init__(self):
        super(MINI, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, 2)),
            ('sigmoid1', nn.Sigmoid())
            ('linear2', nn.Linear(2, 1)),
            ])
    def forward(self, x, weights=None):
        if weights is None:
            self.net.forward(x)
            return super(MINI, self).forward(x)
        else:
            for name, param in self.named_parameters():
                
            x = F.linear(x, weights[net.linear1.weight)
            return torch.sigmoid(F.linear(x, weights['net.linear.weight'], weights['net.linear.bias']))

a = MINI()
x1 = torch.tensor([1.])
x2 = torch.tensor([2.])
#w = torch.zeros(1, requires_grad=True)

#y1 = torch.sigmoid(x1*w)
#grad1 = torch.autograd.grad(y1, w, create_graph=True)
#w1 = w - alpha*grad1[0]
#y1 = torch.sigmoid(x1*w1)
#grad1 = torch.autograd.grad(y1, w, create_graph=True)
#w1 = w - alpha*grad1[0]
#print(y1, grad1, w1)
#
#print(w.grad_fn)
#print(type(w.grad_fn))
#print(dir(w.grad_fn))
##w.grad.zero_()
#
#y2 = torch.sigmoid(x2*w)
#grad2 = torch.autograd.grad(y2, w, create_graph=True)
#w2 = w - alpha*grad2[0]
#print(y2, grad2, w2)
#
#y = (torch.sigmoid(w1*x1) + torch.sigmoid(w2*x2))*.5
#grad = torch.autograd.grad(y, w)
#
#print(y, grad, w)




y1 = a(x1)
grad1 = torch.autograd.grad(y1, a.parameters(), create_graph=True)
for param, grad in zip(a.parameters(), grad1):
    print(param, grad)
w1 = [[param - alpha*grad] for param, grad in zip(a.parameters(), grad1)]
print('w1', w1)

y1_ = a.forward_given_weight(x1, w1)

dot = make_dot(y1_, params=dict(a.named_parameters()))
dot.render()
"""
y2 = a(x2)
grad2 = torch.autograd.grad(y2, a.parameters(), create_graph=True)
w2 = [param - alpha*grad for param, grad in zip(a.parameters(), grad2)]
print('w2', w2)

meta_grads = {name:g for ((name, _), g) 
                    in zip(a.named_parameters(), [grad1, grad2]) }

print(meta_grads)
print(meta_grads.keys())

#grads = {k: sum(d[k] for d in meta_grads) for k in meta_grads[0].keys()}
#print(grads)
#y = 
#make_dot(y, params={w})
"""


"""


x = torch.rand(1)
x0 = torch.rand(1)
x1 = torch.rand(1)
a = Simple()


# Serial
b = copy.deepcopy(a)
loss0 = b(x)**2

optimizer = torch.optim.Adam(b.parameters())
optimizer.zero_grad()
loss0.backward()
optimizer.step()

loss1 = b(x)**2
optimizer = torch.optim.Adam(b.parameters())
optimizer.zero_grad()
loss1.backward()
optimizer.step()


# Parallel
b = copy.deepcopy(a)
c = copy.deepcopy(a)
print('before update abc')
print(a.hidden.weight)
print(b.hidden.weight)
print(c.hidden.weight)
lossb = (b(x0) - torch.tensor([100.]))**2
lossc = (c(x1) - torch.tensor([100.]))**2


#lossb.backward()

#optimizer = torch.optim.Adam(b.parameters())
#optimizer.zero_grad()
#lossb.backward()
#optimizer.step()

optimizer = torch.optim.Adam(c.parameters())
optimizer.zero_grad()
lossc.backward()
optimizer.step()


lossa = .5*(b(x) - torch.tensor([100.]))**2 +  \
    .5*(c(x) - torch.tensor([100.]))**2
optimizer = torch.optim.Adam(a.parameters())
lossa.backward()
optimizer.step()

print('after update abc')
print(a.hidden.weight)
print(b.hidden.weight)
print(c.hidden.weight)















dup = copy.deepcopy(a)
dup.copy_(a)

params = list(a.parameters()) #+ list(dup.parameters())
optimizer = torch.optim.Adam(params)

loss = dup(x)**2

print('before optim')
print(a.hidden.weight)
print(a.hidden.weight.grad_fn)
print(dup.hidden.weight)

print(dup.hidden.weight.requires_grad)
for a_, dup_ in zip(a.parameters(), dup.parameters()):
    print(a_, dup_)


optimizer.zero_grad()
loss.backward()
optimizer.step()

print('\n\nafter optim')
print(a.hidden.weight)
print(dup.hidden.weight)
#for a_, dup_ in zip(a.parameters(), dup.parameters()):
#    print(a_, dup_)
"""
