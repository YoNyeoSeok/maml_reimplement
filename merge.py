import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torchviz import make_dot

from model import Simple
from data.data_gen import Sinewave_data
import numpy as np
from collections import OrderedDict
import argparse
import copy

parser = argparse.ArgumentParser()
#parser.add_argument('--model', default="model")
parser.add_argument('--log_folder', type=str, default="sine")
parser.add_argument('alpha', type=float, nargs='?', default=1e-2)
parser.add_argument('beta', type=float, nargs='?', default=1e-2)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def likelihood_loss_fun(predict_mu, predict_sigma, target):
    pass 

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
    
    def copy_(self, weights):
        for name, param in self.named_parameters():
            param.data = weights[name].data
            



class MAML(nn.Module):
    def __init__(self, model, num_task=1):
        super(MAML, self).__init__()

        self.model = model
        self.num_task = 1

    def forward(self, x):
        pass

    def construct_model(self, args):
        self.model = nn.Sequential(args.model)

def mse(predict, target):
    return torch.mean((predict-target)**2, axis=0)
    

def main():
    data_gen = Sinewave_data()

    inputs, targets = data_gen.gen(n=5, x_min=-5, x_max=5)
#    model = Simple(d_hidden=5)
    config = {}
    config['model'] = OrderedDict([
        ('linear1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('linear3', nn.Linear(40, 1))
        ])
    model = Construct_Sequential(config['model'])
    print(model)
    fast_net = Construct_Sequential(config['model'])
    loss_func = F.mse_loss 
    
    learning_rate = args.beta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    alpha = args.alpha

    batch_size = 25 
    num_step = 1
    num_examples = 5
    max_iter = 70000
    for i in range(max_iter):
        xs = np.zeros((batch_size, num_examples))
        ys = np.zeros((batch_size, num_examples))
        weight0 = {}
        for name, module in model.named_parameters():
           weight0[name] = module 
#        model0 = copy.deepcopy(model)
#        model0.copy_(model)
        losses = torch.zeros((batch_size, num_step, 1))
        meta_losses = torch.zeros((batch_size, 1))

        A = np.random.uniform(1, 5, batch_size).reshape(-1, 1)
        b = np.random.uniform(0, np.pi, batch_size).reshape(-1, 1)

        x = np.random.uniform(-5, 5, batch_size*num_examples).reshape(-1, num_examples)
        y = np.multiply(A, np.sin(x - b))
        
        fast_net.copy_(weight0)
        input = torch.tensor(x.reshape(-1, 1)).float()
        target = torch.tensor(y.reshape(-1, 1)).float()

        predict = fast_net(input)
        print(predict.size(), target.size())

        for j in range(batch_size):
            fast_net.copy_(weight0)
            
#            model_j = copy.deepcopy(model)
#            model_j.copy_(model)
			
            A = np.random.uniform(1, 5, 1)
            b = np.random.uniform(0, np.pi, 1)
            xs[j] = np.random.uniform(-5, 5, num_examples)
            ys[j] = A*np.sin(xs[j]-b)

            input = torch.tensor(xs[j]).reshape(-1, 1).float()
            target = torch.tensor(ys[j]).reshape(-1, 1).float()

            predict = fast_net(input)
            losses[j][0] = loss_func(predict, target)

            grad = torch.autograd.grad(losses[j][0], fast_net.parameters(), create_graph=True)
            weight_ = OrderedDict([ (name, weight0[name] - alpha*grad[idx]) 
                for name, (idx, _) in zip(weight0, enumerate(fast_net.parameters()))
                ])

            for k in range(num_step-1):
                predict = fast_net(input, weight_)
                losses[j][k+1] = loss_func(predict, target)
    
                grad = torch.autograd.grad(losses[j][k+1], fast_net.parameters(), create_graph=True)
                weight_ = OrderedDict([ (name, weight_[name] - alpha*grad[idx]) 
                    for name, (idx, _) in zip(weight_, enumerate(fast_net.parameters()))
                    ])

            meta_predict = model.forward(input, weight_)
            meta_losses[j] = loss_func(meta_predict, target) / batch_size

        loss = torch.mean(losses)
        meta_loss = torch.sum(meta_losses)
        
        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
            log_file.write("%5d\t%10.4f\n"%(i, meta_loss.item()))

        if i % 1000 == 0:
            #print("%4d, loss=%.4f"%(i, meta_loss))
            print("%4d, preloss=%.4f \t postloss=%.4f"%(i, loss, meta_loss))
            torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, i))
#            torch.save(model, "logs/sine/%05d.pt"%(i))
            dot = make_dot(meta_loss)
            dot.render()
        
        optimizer.zero_grad()
        meta_loss.backward()
#        for name, param in model.named_parameters():
#            print(param.grad)
        optimizer.step()


if __name__ == '__main__':
    main()
