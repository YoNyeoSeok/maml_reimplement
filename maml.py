import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

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
    ('flatten1', Change_view((-1, 64)))
    ])
dict3 = OrderedDict([
    ('fc1', nn.Linear(64, 5))
    ])

print(dict2.update(dict3))
print(dict.update(dict2))

model = Construct_Sequential(dict)
fast_net = Construct_Sequential(dict)
feature = Construct_Sequential(dict)
#print('parameters', model._parameters)
print(model)
#x = torch.rand((1, 1, 28, 28))
x = torch.rand((2, 1, 28, 28))
print(x.shape)
y = model(x)
print(y.shape)

import os, random
from PIL import Image
import numpy as np
data_folder = './data/omniglot_resized'
character_folders = [os.path.join(data_folder, family, character) \
        for family in os.listdir(data_folder) \
        if os.path.isdir(os.path.join(data_folder, family)) \
        for character in os.listdir(os.path.join(data_folder, family))]
random.seed(1)
random.shuffle(character_folders)
num_val = 100
num_train = 1200 - num_val

metatrain_character_folders = character_folders[:num_train]
metaval_character_folders = character_folders[num_train+num_val:]
rotations = [0, 90, 180, 270]

batch_size = 8
alpha = .4
num_classes = 5
num_examples = 1

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

max_iter = 20
for i in range(max_iter):
    characters = [random.sample(metatrain_character_folders, num_classes) 
            for i in range(batch_size)]
    imagePath_label = [[(os.path.join(character, image), j) \
                for j, character in enumerate(characters[i])
                for image in random.sample(os.listdir(character), num_examples*2)] \
            for i in range(batch_size)]
    #print(imagePath_label[0])
    
    imagePath_label = [[[i[j*2] for j in p]+[i[j*2+1] for j in p]
            for p in [np.random.permutation(num_classes)]][0]
        for i in imagePath_label]
    #print(imagePath_label[0])
    
    imagePaths = [[imagePath_label[i][j][0] for j in range(num_classes*num_examples*2)]
            for i in range(batch_size)]
    labels = [[imagePath_label[i][j][1] for j in range(num_classes*num_examples*2)]
            for i in range(batch_size)]
    
    images = np.array([[list(Image.open(imagePaths[i][j]).rotate(random.choice(rotations)).getdata())                            # (N * C * K, 28*28)
            for j in range(num_classes*num_examples*2)]
        for i in range(batch_size)]) / 255.
    images = 1-images
    labels = np.array(labels)
    
    input_image_shape = (-1, 1, 28, 28)
    input = torch.tensor(images[:, :num_classes*num_examples]).reshape(input_image_shape).float()
    inputb = torch.tensor(images[:, num_classes*num_examples:]).reshape(input_image_shape).float()
     
    target = torch.tensor(labels[:, :num_classes*num_examples]).reshape(-1,).long()
    targetb = torch.tensor(labels[:, num_classes*num_examples:]).reshape(-1,).long()
    
    weight0 = {}
    for name, param in model.named_parameters():
        weight0[name] = param
    fast_net._copy(weight0)
    
    predict = fast_net(input)
    #print(torch.argmax(torch.softmax(predict, dim=1), dim=1))
    #print(target)
    print('preaccuracy', (torch.argmax(torch.softmax(predict, dim=1), dim=1)==target).float().mean())
    
    loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fun(predict, target)
    loss = torch.mean(losses)
    
    weights = []
    for j in range(batch_size*num_classes*num_examples):
        grad = torch.autograd.grad(losses[j], fast_net.parameters(), create_graph=True)
        weights.append(OrderedDict([ (name, weight0[name] - alpha*grad[idx])
            for name, (idx, _) in zip(weight0, enumerate(fast_net.parameters())) ]))
    
    predictb = torch.cat([model(inputb[j:j+1], weight) for j, weight in enumerate(weights)])
    lossesb = loss_fun(predictb, targetb)
    meta_loss = torch.mean(lossesb)
    #print(torch.argmax(torch.softmax(predictb, dim=1), dim=1))
    #print(targetb)
    print('postaccuracy', (torch.argmax(torch.softmax(predictb, dim=1), dim=1)==targetb).float().mean())
    
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

#print(labels)
#print(target, targetb)



#for name, parameter in model.named_parameters():
#    print(name, parameter)
#    weights_dict[name] = parameter


