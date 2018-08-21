import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
#from torchviz import make_dot

from model import Simple
from data.data_gen import Sinewave_data
import numpy as np
from collections import OrderedDict
import argparse
import copy
import os
import random

from PIL import Image

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
            
from maml import Construct_Sequential, Change_view



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
    # Sinusoidal data
    data_gen = Sinewave_data()
    inputs, targets = data_gen.gen(n=5, x_min=-5, x_max=5)

    # Omniglot data
    data_folder = './data/omniglot_resized'
    character_folders = [os.path.join(data_folder, family, character) \
            for family in os.listdir(data_folder) \
            if os.path.isdir(os.path.join(data_folder, family)) \
            for character in os.listdir(os.path.join(data_folder, family))]
    #print(np.array(character_folders).shape)
    random.seed(1)
    random.shuffle(character_folders)
    num_val = 100
    num_train = 1200 - num_val

    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:num_train+num_val]          # for test
    metaval_character_folders = character_folders[num_train+num_val:]                   # validation in training
    rotations = [0, 90, 180, 270]


    config = {}
    # model for Sinusoidal
    config['model'] = OrderedDict([
        ('linear1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('linear3', nn.Linear(40, 1))
        ])
    model = Construct_Sequential(config['model'])
    # model for Omniglot
    config['features'] = OrderedDict([
		('conv1', nn.Conv2d(1, 64, (3, 3))),
	    ('bn1', nn.BatchNorm2d(64, track_running_stats=False)),
	    ('relu1', nn.ReLU()),
	    ('pool1', nn.MaxPool2d((2, 2))),
	    ('conv2', nn.Conv2d(64, 64, (3, 3))),
	    ('bn2', nn.BatchNorm2d(64, track_running_stats=False)),
	    ('relu2', nn.ReLU()),
	    ('pool2', nn.MaxPool2d((2, 2))),
	    ('conv3', nn.Conv2d(64, 64, (3, 3))),
	    ('bn3', nn.BatchNorm2d(64, track_running_stats=False)),
	    ('relu3', nn.ReLU()),
	    ('pool3', nn.MaxPool2d((2, 2))),
	    ])  
    config['view'] = OrderedDict([
        ('view', Change_view((-1, 64)))
        ])
    config['classifier'] = OrderedDict([
        ('fc1', nn.Linear(64, 5))
        ])
    config['model'] = OrderedDict(list(config['features'].items()) 
            + list(config['view'].items()) + list(config['classifier'].items()))
    model = Construct_Sequential(config['model'])
    print(model)
    fast_net = Construct_Sequential(config['model'])
	
	# loss for Sinusoidal
    loss_func = F.mse_loss 
    loss_func = torch.nn.MSELoss()
	# loss for Omniglot
    loss_func = F.cross_entropy
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    
    learning_rate = args.beta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    alpha = args.alpha

    # sinusiodal
    batch_size = 25 
    num_step = 1
    num_examples = 5
    max_iter = 70000
    # omniglot iteration
    max_iter = 40000
    num_classes = 20    # 20-way
    num_examples = 1    # 1-shot
    batch_size = 16     # for 20-way 1-shot
    alpha = .1          # for 20-way 1-shot
    num_step = 5        # for 20-way 1-shot

    num_classes = 5     # 5-way
    num_examples = 1    # 1-shot
    batch_size = 32     # for 5-way 1-shot
    alpha = .4          # for 5-way 1-shot
    num_step = 1        # for 5-way 1-shot
    for i in range(max_iter):
        # for sinusoidal
        xs = np.zeros((batch_size, num_examples))
        ys = np.zeros((batch_size, num_examples))

        # for omniglot
        input_characters = random.sample(metatrain_character_folders, batch_size*num_classes)   # (N * C, 1)
        imagePath_label = [(os.path.join(character, image), j) \
                for i in range(batch_size) \
                for j, character in enumerate(input_characters[i*num_classes:(i+1)*num_classes]) \
                for image in random.sample(os.listdir(character), num_examples)]                # (N * C * K, 1)

        imagePaths = [imagePath_label[i][0] for i in range(batch_size*num_classes*num_examples)]
        labels = [imagePath_label[i][1] for i in range(batch_size*num_classes*num_examples)]
        #print(np.array(imagePath_label).shape)

        images = np.array([list(Image.open(imagePaths[i]).getdata())                            # (N * C * K, 28*28)
            for i in range(batch_size*num_classes*num_examples)]) / 255.
        #print(images.shape)
        input_image_shape = (-1, 1, 28, 28)

        input = torch.tensor(images).reshape(input_image_shape).float()
        target = torch.tensor([labels[i] for i in range(batch_size*num_classes*num_examples)]).long()
        #print(input.shape)
        #print(target.shape)

        weight0 = {}
        for name, module in model.named_parameters():
           weight0[name] = module
        fast_net._copy(weight0)

        losses = torch.zeros((batch_size, num_step, 1))
        meta_losses = torch.zeros((batch_size, 1))

        predict = fast_net(input)
        #print(predict.shape)
        #print(loss_func)
        losses = loss_func(predict, target)
        #print(losses)

        grads = [torch.autograd.grad(loss, fast_net.parameters(), create_graph=True) 
                for loss in losses]
        weights_ = [ OrderedDict([ (name, weight0[name] - alpha*grad[idx])
                        for name, (idx, _) in zip(weight0, enumerate(fast_net.parameters())) ])
                    for grad in grads]
        #for name in weight0:
        #    print(name)
        """
        fast_net._copy(weights

        predict_ = []
        losses_ = []
        for k in range(num_step-1):
            predict_.append(fast_net(input))
            losses_.append(loss_func(predict_[-1], target))
            
            grads = [torch.autograd.grad(loss, fast_net.parameters(), create_graph=True) 
                    for loss in losses_[-1]]
            weights_ = [ OrderedDict([ (name, weight0[name] - alpha*grad[idx])
                            for name, (idx, _) in zip(weight0, enumerate(fast_net.parameters())) ])
                        for grad in grads]
        """ 
        loss = torch.mean(losses)
        meta_predicts = [model.forward(input, weight_) for weight_ in weights_]
        meta_loss = torch.mean(torch.cat([loss_func(meta_predict, target) for meta_predict in meta_predicts]))
        """ 
        for j in range(batch_size):
            fast_net._copy(weight0)
            
		    # for sinusoidal	
            A = np.random.uniform(1, 5, 1)
            b = np.random.uniform(0, np.pi, 1)
            xs[j] = np.random.uniform(-5, 5, num_examples)
            ys[j] = A*np.sin(xs[j]-b)

            input = torch.tensor(xs[j]).reshape(-1, 1).float()
            target = torch.tensor(ys[j]).reshape(-1, 1).float()

            # for omniglot
#            sampled_characters[j] = random.samples(metatrain_character_folders, num_examples)
#            lable_and_imagepaths = [(i, image_path) for zip()]

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
        
"""
        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
            log_file.write("%5d\t%10.4f\n"%(i, meta_loss.item()))

        if i % 100 == 0:
            #print("%4d, loss=%.4f"%(i, meta_loss))
            print("%4d, preloss=%.4f \t postloss=%.4f"%(i, loss, meta_loss))
            torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, i))
#            torch.save(model, "logs/sine/%05d.pt"%(i))
            #dot = make_dot(meta_loss)
            #dot.render()
        
        optimizer.zero_grad()
        meta_loss.backward()
#        for name, param in model.named_parameters():
#            print(param.grad)
        optimizer.step()


if __name__ == '__main__':
    main()
