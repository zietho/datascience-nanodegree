
import json
import matplotlib.pyplot as plt
import numpy as np
import time


import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import models


class Network: 
    #constructor 
    def __init__(self, model, args):
        self.model = model, 
        self.input_size = model.classifier[0].in_features
        self.data_dir = args.data_dir if hasattr(args,'data_dir') else '/'
        self.save_dir = args.save_dir if hasattr(args,'save_dir') else '/'
        self.epochs = args.epochs if hasattr(args,'epochs') else 5
        self.learning_rate = args.learning_rate if hasattr(args,'learning_rate') else 0.001
        self.device = args.device if hasattr(args,'device') else 'cpu'
        #TODO figure out how to measure this!
        self.output_size = 3# ? if categories are only used in predict - how can we get this info ? 
        #TODO figure out what is meant by e.g., 512 ? 
        self.hidden_layers = args.hidden_layers if hasattr(args, 'hidden_layers') else [3]

    def set_model(self, model):
        self.model = model

    def build_classifier(self):
        # freeze params of pretrained network
        for param in self.model.parameters():
            param.requires_grad = False
        
        modules = []
        
        # init input layer
        modules.append(nn.Linear(self.input_size, self.hidden_size[0]))
        
        # add hidden layers
        for index, size in enumerate(self.hidden_size):
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.2))
            output = self.hidden_size[index+1] if index < (len(self.hidden_size)-1) else self.output_size
            modules.append(nn.Linear(size, output))
        
        # add activation function
        modules.append(nn.LogSoftmax(dim=1))
                    
        # build classifier by unpacking modules into a nn.Sequential 
        classifier = nn.Sequential(*modules)

        # set classifier            
        self.model.classifier = classifier
    
    def train_model(self, train_dataloader, validation_dataloader, criterion, optimizer, print_info_every=50):
        # move model to gpu
        self.model.to(self.device)
        
        steps = 0

        for e in range(self.epochs):
            running_loss = 0
            for images, labels in train_dataloader:
                self.steps +=1
                # Move input and label tensors to the GPU
                images, labels = images.to(self.device), labels.to(self.device)

                # reset 
                optimizer.zero_grad()

                log_ps = self.model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if steps % self.print_info_everyvery == 0:
                    accuracy = 0 
                    test_loss = 0 
                    self.model.eval()
                    
                    with torch.no_grad():
                        # validation pass here
                        for images, labels in validation_dataloader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            log_ps = self.model(images)
                            test_loss += criterion(log_ps,labels).item()
                            ps = torch.exp(log_ps)

                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print("epoch: {}/{}".format(e+1,self.epochs),
                        "Train loss: {:.3f}..".format(running_loss/self.print_info_every),
                        "Test loss: {:.3f}..".format(test_loss/len(validation_dataloader)),
                        "f'Accuracy: {:.3f}".format(accuracy/len(validation_dataloader)))             

                    #reset 
                    running_loss=0
                    self.model.train()   

    def validate_model(self, criterion, dataloader):
        # Put model in inference mode
        self.model.eval()
        accuracy = 0
        loss = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model.forward(images)
                loss += criterion(output, labels).item()
                
                # exp of softmax 
                ps = torch.exp(output).data
                # TODO - check this line!? 
                equality = (labels.data == ps.max(1)[1])
                
                # calc mean of predictions
                accuracy += equality.type_as(torch.FloatTensor()).mean()

        validation_loss = loss/len(dataloader)
        validation_accuracy = accuracy/len(dataloader)

        return validation_loss, validation_accuracy

    def save_checkpoint(self, checkpoint_name, class_to_idx): 
        # create checkpoint
        checkpoint = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size, 
            'class_to_idx': self.image_data['train'].class_to_idx,
            'epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        # save checkpoint
        torch.save(checkpoint, checkpoint_name)                    


    def load_checkpoint(self, model, filepath):
        checkpoint = torch.load(filepath)
        
        # get meta infos
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.output_size = checkpoint['output_size']
        
        # build and set classifier such that the loaded checkpoint can be applied correctly
        model.classifier = self.build_classifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
        # set and load optimizer
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print('input_size: {}'.format(checkpoint['input_size']))
        print('output_size: {}'.format(checkpoint['output_size']))
        print('epochs {}'.format(checkpoint['epochs']))
        
        self.model = model 
        self.optimizer = optimizer
        

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