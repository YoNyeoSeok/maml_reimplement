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
parser.add_argument('--datasource', type=str, nargs='?', default="sinusoidal", choices=["sinusoidal", "omniglot"])
parser.add_argument('--batch_size', type=int, nargs='?', default=25)
parser.add_argument('--num_examples', type=int, nargs='?', default=5)
parser.add_argument('--num_classes', type=int, nargs='?', default=5)
parser.add_argument('--num_updates', type=int, nargs='?', default=1)
parser.add_argument('--log_folder', type=str)
parser.add_argument('--alpha', type=float, nargs='?', default=1e-2)
parser.add_argument('--beta', type=float, nargs='?', default=1e-2)
parser.add_argument('--max_iter', type=int, nargs='?', default=15000)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from maml import Construct_Sequential, Change_view

def main():
    # Sinusoidal data
    classification = False
    if args.datasource == "sinusoidal":
        data_gen = Sinewave_data()
        inputs, targets = data_gen.gen(n=5, x_min=-5, x_max=5)
    # Omniglot data
    elif args.datasource == "omniglot":
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
    if args.datasource == "sinusoidal":
        config['model'] = OrderedDict([
            ('linear1', nn.Linear(1, 40)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(40, 1))
            ])
        model = Construct_Sequential(config['model'])
    # model for Omniglot
    elif args.datasource == "omniglot":
        config['features'] = OrderedDict([
    		('conv1', nn.Conv2d(1, 64, 3, 2)),
    	    ('bn1', nn.BatchNorm2d(64, track_running_stats=False)),
    	    ('relu1', nn.ReLU()),
            #('pool1', nn.MaxPool2d((2, 2))),
    	    ('conv2', nn.Conv2d(64, 64, 3, 2)),
    	    ('bn2', nn.BatchNorm2d(64, track_running_stats=False)),
    	    ('relu2', nn.ReLU()),
            #('pool2', nn.MaxPool2d((2, 2))),
    	    ('conv3', nn.Conv2d(64, 64, 3, 2)),
    	    ('bn3', nn.BatchNorm2d(64, track_running_stats=False)),
    	    ('relu3', nn.ReLU()),
            #('pool3', nn.MaxPool2d((2, 2))),
    	    ('conv4', nn.Conv2d(64, 64, 3, 2, 1)),
    	    ('bn4', nn.BatchNorm2d(64, track_running_stats=False)),
    	    ('relu4', nn.ReLU()),
            #('pool4', nn.MaxPool2d((2, 2))),
    	    ])  
        config['view'] = OrderedDict([
            ('view', Change_view((-1, 64)))
            ])
        config['classifier'] = OrderedDict([
            ('fc1', nn.Linear(64, args.num_classes))
            ])
        config['model'] = OrderedDict(list(config['features'].items()) 
                + list(config['view'].items()) + list(config['classifier'].items()))
        model = Construct_Sequential(config['model'])
#    for name, param in model.named_parameters():
#        print(name, param.shape)
        for (name, module) in model._modules.items():
            #print(name, module)
            if isinstance(module, nn.Conv2d):
            #if hasattr(module, 'weight') and not 'bn' in name:
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.ones_(module.weight)
                #print(module.weight)
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight)
                torch.nn.init.ones_(module.weight)
                #print(module.weight)
    model = model.to(device)
    #print(model)
#    print('model', torch.cuda.memory_allocated(torch.cuda.current_device()))
#    fast_net = Construct_Sequential(config['model']).to(device)
	
	# loss for Sinusoidal
    if args.datasource == "sinusoidal":
        loss_func = F.mse_loss 
        loss_func = torch.nn.MSELoss(reduction='none')
	# loss for Omniglot
    elif args.datasource == "omniglot":
        loss_func = F.cross_entropy
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.Adam(model.parameters(), lr=args.beta)

    validation = False
    for i in range(args.max_iter):
        # for sinusoidal
        if args.datasource == "sinusoidal":
            A = torch.empty(args.batch_size, 1).uniform_(1, 5).to(device)
            b = torch.empty(args.batch_size, 1).uniform_(0, np.pi).to(device)
            inputs = torch.empty(args.batch_size, args.num_examples).uniform_(-5, 5).to(device)
            targets = A*torch.sin(inputs - b).to(device)
            inputbs = torch.empty(args.batch_size, args.num_examples).uniform_(-5, 5).to(device)
            targetbs = A*torch.sin(inputbs - b).to(device)

        # for omniglot
        elif args.datasource == "omniglot":
            if validation:
                input_characters = [random.sample(metaval_character_folders, args.num_classes) for i in range(args.batch_size)]     # (N, C)
            else:
                input_characters = [random.sample(metatrain_character_folders, args.num_classes) for i in range(args.batch_size)]     # (N, C)
            imagePath_label = [[(os.path.join(character, image), j) \
                        for j, character in enumerate(input_characters[i]) \
                        for image in random.sample(os.listdir(character), args.num_examples*2)] \
                    for i in range(args.batch_size)]                                                                         # (N, C*K*2)
            
            imagePath_label = [[[i[j*2] for j in p]+[i[j*2+1] for j in p] 
                        for p in [np.random.permutation(args.num_classes)]][0]
                    for i in imagePath_label]
    
            imagePaths = [[imagePath_label[i][j][0] for j in range(args.num_classes*args.num_examples*2)]
                for i in range(args.batch_size)]
            labels = [[imagePath_label[i][j][1] for j in range(args.num_classes*args.num_examples*2)]
                for i in range(args.batch_size)]
            
            images = np.array([[list(Image.open(imagePaths[i][j]).rotate(random.choice(rotations)).getdata())                            # (N, C*K*2, 28*28)
                    for j in range(args.num_classes*args.num_examples*2)]
                for i in range(args.batch_size)]) / 255.
            images = 1-images
            labels = np.array(labels)
            input_image_shape = (-1, 1, 28, 28)
    
            inputs = torch.tensor(images[:, :args.num_classes*args.num_examples]).float().to(device)
            inputbs = torch.tensor(images[:, args.num_classes*args.num_examples:]).float().to(device)
    
            targets = torch.tensor(labels[:, :args.num_classes*args.num_examples]).long().to(device)
            targetbs = torch.tensor(labels[:, args.num_classes*args.num_examples:]).long().to(device)
        
        weight0 = {}
        for name, pram in model.named_parameters():
           weight0[name] = pram 
        #fast_net._copy(weight0)

        outputs, _outputs = [], [[] for i in range(args.batch_size)]
        losses, _losses = [], [[] for i in range(args.batch_size)]
        if classification:
            accuracies, _accuracies = [], [[] for i in range(args.batch_size)]

        for j in range(args.batch_size):
            if args.datasource == "sinusoidal":
                input = inputs[j].reshape(args.num_examples, -1)
                inputb = inputbs[j].reshape(args.num_examples, -1)
                target = targets[j].reshape(args.num_examples, -1)
                targetb = targetbs[j].reshape(args.num_examples, -1)
            if args.datasource == "omniglot":
                #                input = inputs[j].reshape(input_image_shape)
#                inputb = inputbs[j].reshape(input_image_shape)
#                target = targets[j].reshape(args.num_classes*args.num_examples,)
#                targetb = targetbs[j].reshape(args.num_classes*args.num_examples,)
                input = torch.tensor(np.ones((5, 1, 28, 28)), dtype=torch.float).to(device)
                inputb = torch.tensor(np.ones((5, 1, 28, 28)), dtype=torch.float).to(device)
                target = torch.tensor(np.arange(5)).to(device)
                targetb = torch.tensor(np.arange(5)).to(device)
            
            output = model(input)
            outputs.append(output)
            loss_ = loss_func(outputs[-1], target) / args.num_examples
            loss = torch.sum(loss_, 0, True)
            losses.append(loss)
            if classification:
                logits = output
                prob = torch.softmax(logits, dim=1)
                predict = torch.argmax(prob, dim=1)
                accuracy = (predict == target).float().mean(0, True)
                accuracies.append(accuracy)

            param = model.parameters()
            grads = torch.autograd.grad(losses[-1], param, create_graph=True) 
            weight_ = OrderedDict([ (name, weight0[name] - args.alpha*grad)
                        for name, grad in zip(weight0, grads) ])

            _output = model.forward(inputb, weight_)
            _outputs[j].append(_output)
            _loss_ = loss_func(_outputs[j][-1], targetb) / args.num_examples
            _loss = torch.sum(_loss_, 0, True)
            _losses[j].append(_loss)
            if classification:
                _logits = _output
                _prob = torch.softmax(_logits, dim=1)
                _predict = torch.argmax(_prob, dim=1)
                _accuracy = (_predict == targetb).float().mean(0, True)
                _accuracies[j].append(_accuracy)

            for k in range(args.num_updates-1):
                param = list([w for k, w in weight_.items()])
                grads = torch.autograd.grad(_losses[j][-1], param, create_graph=True) 
                weight_ = OrderedDict([ (name, weight_[name] - args.alpha*grad)
                            for name, grad in zip(weight_, grads) ])

                _output = model.forward(inputb, weight_)
                _outputs[j].append(_output)
                _loss_ = loss_func(_outputs[j][-1], targetb) / args.num_examples
                _loss = torch.sum(_loss_, 0, True)
                _losses[j].append(_loss)
                if classification:
                    _logits = _output
                    _prob = torch.softmax(_logits, dim=1)
                    _predict = torch.argmax(_prob, dim=1)
                    _accuracy = (_predict == targetb).float().mean(0, True)
                    _accuracies[j].append(_accuracy)
        total_losses = torch.cat(losses)
        total_loss = torch.mean(total_losses)
        _total_losses = torch.stack([torch.cat(_loss) for _loss in _losses])
        _total_loss = torch.mean(_total_losses, dim=0)
        if classification:
            tatal_accuracies = torch.cat(accuracies)
            total_accuracy = torch.mean(tatal_accuracies)
            _tatal_accuracies = torch.stack([torch.cat(_accuracy) for _accuracy in _accuracies])
            _total_accuracy = torch.mean(_tatal_accuracies, dim=0)
#            predicts = torch.cat(predicts)
#            meta_predicts = torch.cat(meta_predicts)
#            accuracy = (predicts == targets).float().mean()
#            meta_accuracy = (meta_predicts == targetbs).float().mean()
        
        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
            log_file.write("%5d\t%10.4f\t"%(i, total_loss.item()))
            for k in range(args.num_updates):
                log_file.write("%10.4f\t"%(_total_loss[k].item()))
            if classification:
                log_file.write("%10.4f\t"%(total_accuracy.item()))
                for k in range(args.num_updates):
                    log_file.write("%10.4f\t"%(_total_accuracy[k].item()))
                
            log_file.write("\n")


        if i % 100 == 0:
            #print("%4d, loss=%.4f"%(i, meta_loss))
            if classification:
                #print(predicts[-1], meta_predicts[-1], targets[-1], targetbs[-1])
                #print(loss_)
                #print(outputs)
                #print(meta_outputs)
                #print("%4d, preloss=%.4f \t postloss=%.4f"%(i, total_loss, _total_loss[args.num_updates-1]))
                print(total_losses)
                print("%4d, preaccuracy=%.4f \t postaccuracy=%.4f"%(i, total_accuracy, _total_accuracy[args.num_updates-1]))
            else:
                print("%4d, preloss=%.4f \t postloss=%.4f"%(i, total_loss, _total_loss[args.num_updates-1]))
            torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, i))
        optimizer.zero_grad()

        if validation:
            validation = False
        if i % 500 == 0:
            validation = True
        else:
            _total_loss[args.num_updates-1].backward()
#        for name, param in model.named_parameters():
#            print(param.grad)
            optimizer.step()
        
        # for validation


if __name__ == '__main__':
    main()
