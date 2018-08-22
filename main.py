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
parser.add_argument('--log_folder', type=str, default="omniglot5way")
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
    classification = True
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
    model = Construct_Sequential(config['model']).to(device)
    for name, param in model.named_parameters():
        print(name, param.shape)
    print(model)
    print('model', torch.cuda.memory_allocated(torch.cuda.current_device()))
    fast_net = Construct_Sequential(config['model']).to(device)
	
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
    batch_size = 8#16#32     # for 5-way 1-shot
    alpha = .4          # for 5-way 1-shot
    num_step = 1        # for 5-way 1-shot

    validation = False
    for i in range(max_iter):
        # for sinusoidal
        xs = np.zeros((batch_size, num_examples))
        ys = np.zeros((batch_size, num_examples))

        # for omniglot
        if validation:
            input_characters = [random.sample(metaval_character_folders, num_classes) for i in range(batch_size)]     # (N, C)
        else:
            input_characters = [random.sample(metatrain_character_folders, num_classes) for i in range(batch_size)]     # (N, C)
        imagePath_label = [[(os.path.join(character, image), j) \
                    for j, character in enumerate(input_characters[i]) \
                    for image in random.sample(os.listdir(character), num_examples*2)] \
                for i in range(batch_size)]                                                                         # (N, C*K*2)
        #print(imagePath_label)
        
        imagePath_label = [[[i[j*2] for j in p]+[i[j*2+1] for j in p] 
                    for p in [np.random.permutation(num_classes)]][0]
                for i in imagePath_label]
        #print(imagePath_label)

        imagePaths = [[imagePath_label[i][j][0] for j in range(num_classes*num_examples*2)]
            for i in range(batch_size)]
        labels = [[imagePath_label[i][j][1] for j in range(num_classes*num_examples*2)]
            for i in range(batch_size)]
        #print(labels)
        #print(np.array(imagePath_label).shape)

        images = np.array([[list(Image.open(imagePaths[i][j]).rotate(random.choice(rotations)).getdata())                            # (N * C * K, 28*28)
                for j in range(num_classes*num_examples*2)]
            for i in range(batch_size)]) / 255.
        labels = np.array(labels)
        #print(images.shape)
        input_image_shape = (-1, 1, 28, 28)

        input = torch.tensor(images[:, :num_classes*num_examples]).float().to(device)
        inputb = torch.tensor(images[:, num_classes*num_examples:]).float().to(device)

        target = torch.tensor(labels[:, :num_classes*num_examples]).long().to(device)
        targetb = torch.tensor(labels[:, num_classes*num_examples:]).long().to(device)
#        target = torch.tensor([labels[i] for i in range(batch_size*num_classes*num_examples)]).long().to(device)
#        targetb = torch.tensor([labels[batch_size*num_classes:][i] for i in range(batch_size*num_classes*num_examples)]).long().to(device)
        #print(input.shape)
        #print(target.shape)
        #print('input, target', torch.cuda.memory_allocated(torch.cuda.current_device()))
        weight0 = {}
        for name, pram in model.named_parameters():
           weight0[name] = pram 
        #fast_net._copy(weight0)

        #losses = torch.zeros((batch_size, num_step, 1))
        #meta_losses = torch.zeros((batch_size, 1))

        #predict = fast_net(input.reshape(input_image_shape))
        #losses = loss_func(predict, target.reshape(-1))
        #loss = torch.mean(losses)
        #print(torch.softmax(predict, dim=1))
        #print(torch.argmax(torch.softmax(predict, dim=1), dim=1))
        #print(target)
        #print(predict.shape)
        #print(loss_func)
        #print(losses)
        #print('fast_net', torch.cuda.memory_allocated(torch.cuda.current_device()))

        #weights_= []
        meta_predicts = []
        meta_losses = []
        for j in range(batch_size):
            fast_net._copy(weight0)
            predict = fast_net(input[j].reshape(input_image_shape))
            losses = loss_func(predict, target[j].reshape(-1))
            loss = torch.mean(losses)
            #print(j, loss)

            grad = torch.autograd.grad(loss, fast_net.parameters(), create_graph=True) 
            weight_ = OrderedDict([ (name, weight0[name] - alpha*grad[idx])
                        for name, (idx, _) in zip(weight0, enumerate(fast_net.parameters())) ])
            #weights_.append(weight_)
            meta_predict = model.forward(inputb[j].reshape(input_image_shape), weight_)
            meta_loss = torch.mean(loss_func(meta_predict, targetb[j].reshape(-1)), 0, True)
            #print(j, meta_loss)

            meta_predicts.append(meta_predict.reshape(-1, num_examples*num_classes, num_classes))
            meta_losses.append(meta_loss)
        meta_predicts = torch.cat(meta_predicts)
        meta_losses = torch.cat(meta_losses)
        meta_loss = torch.mean(meta_losses)
        #print(meta_predicts.shape)
        #print(meta_loss)
            #meta_predict = torch.cat([model.forward(inputb[j:j+num_classes*num_examples], weight_) for j, weight_ in enumerate(weights_)])
            #meta_loss = torch.mean(loss_func(meta_predicts, targetb))
        #print('grads', torch.cuda.memory_allocated(torch.cuda.current_device()))
        #meta_predicts = torch.cat([model.forward(inputb[j:j+num_classes*num_examples], weight_) for j, weight_ in enumerate(weights_)])
        #meta_loss = torch.mean(loss_func(meta_predicts, targetb))
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
        #img = input[0][0].cpu().data.numpy()*255
        #print(img.astype(int))
        #img = inputb[0][0].cpu().data.numpy()*255
        #print(img.astype(int))
        #Image.fromarray((input[0][0].data.cpu().numpy()*255).astype(int)).show()
        #print(torch.softmax(meta_predicts, dim=1))
        idx = np.random.randint(batch_size*num_classes*num_examples)
        #print(torch.argmax(torch.softmax(meta_predicts, dim=1), dim=1))
        #print(targetb)
        #print(torch.argmax(torch.softmax(model.forward(inputb[idx:idx+1], weights_[idx]), dim=1), dim=1), targetb[idx:idx+1])
        #print(meta_predicts.shape, targetb.shape)
        #print(loss_func(meta_predicts, targetb))
        #print(i)
        #print([(meta_predict, target) for meta_predict in meta_predicts])

        if classification == True:
            #print('softmax', torch.softmax(meta_predicts, dim=1))
            #print(torch.argmax(torch.softmax(meta_predicts, dim=1), dim=1), targetb)
            accuracy = (torch.argmax(torch.softmax(predict, dim=1), dim=1) == target).float().mean()
            meta_accuracy = (torch.argmax(torch.softmax(meta_predicts, dim=1), dim=1) == targetb).float().mean()
            

#            accuracy = torch.mean(torch.cat([ )for meta_predict in zip(meta_predicts, target)]))
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
        #print("%4d, preaccuracy=%.4f \t postaccuracy=%.4f"%(i, accuracy, meta_accuracy))
        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
            log_file.write("%5d\t%10.4f\n"%(i, meta_loss.item()))

        if i % 100 == 0:
            #print("%4d, loss=%.4f"%(i, meta_loss))
            if classification:
                print("%4d, preaccuracy=%.4f \t postaccuracy=%.4f"%(i, accuracy, meta_accuracy))
            else:
                print("%4d, preloss=%.4f \t postloss=%.4f"%(i, loss, meta_loss))
            torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, i))
#            torch.save(model, "logs/sine/%05d.pt"%(i))
            #dot = make_dot(meta_loss)
            #dot.render()
        optimizer.zero_grad()

        if validation:
            validation = False
        if i % 500 == 0:
            validation = True
        else:
            meta_loss.backward()
#        for name, param in model.named_parameters():
#            print(param.grad)
            optimizer.step()
        
        # for validation


if __name__ == '__main__':
    main()
