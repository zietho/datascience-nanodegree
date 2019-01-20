
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image

class Network: 
    #constructor 
    def __init__(self, args):
        self.save_dir = args.save_dir if hasattr(args,'save_dir') else '/'
        self.epochs = args.epochs if hasattr(args,'epochs') else 5
        self.epochs = args.epochs if hasattr(args,'epochs') else 5

    def build_classifier(self, input_size, hidden_size, output_size):
        modules = []
        
        # init input layer
        modules.append(nn.Linear(input_size, hidden_size[0]))
        
        # add hidden layers
        for index, size in enumerate(hidden_size):
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.2))
            output = hidden_size[index+1] if index < (len(hidden_size)-1) else output_size
            modules.append(nn.Linear(size, output))
        
        # add activation function
        modules.append(nn.LogSoftmax(dim=1))
                    
        #  build classifier by unpacking modules into a nn.Sequential 
        classifier = nn.Sequential(*modules)
                    
        return classifier

# utility class to load pretrained models from torchvision.models 
class TorchvisionModels: 
    # constructor
    def __init__(self, architecture):
        self.architecture = architecture

    def load(self): 
        if hasattr(models, self.architecture):
            # load model from models
            model = models.getattr(self.architecture)(pretrained=True)
            return model
        else: 
            return UnknownModelError('the entered model is not part of torchvision.models') 


class UnknownModelError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message