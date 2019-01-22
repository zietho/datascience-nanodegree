import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import models

class Network: 
    #constructor 
    def __init__(self, args): 
        # init hyperparameters
        self.epochs = args.epochs if hasattr(args,'epochs') else 5
        self.learning_rate = args.learning_rate if hasattr(args,'learning_rate') else 0.001
        self.device = args.device if hasattr(args,'device') else 'cpu'
        self.print_info_every = args.print_info_every if hasattr(args, 'print_info_every') else 50
        self.architecture = args.architecture if hasattr(args, 'architecture') else None
        self.hidden_units = range(args.hidden_units) if hasattr(args, 'hidden_units') else 512
        self.output_size = args.output_size if hasattr(args, 'output_size') else 102
        self.logging_level = getattr(logging, args.logging_level) if hasattr(args, 'logging_level') else getattr(logging, 'WARNING')

        # set log level
        logging.basicConfig(stream=sys.stderr, level=self.logging_level)

        # fetch selected model by architecture
        if self.architecture is not None:  
            try:
                self.model = TorchvisionModels(self.architecture).load()
            except UnknownModelError as error:
                logging.error(error)
        else: 
            logging.warning('No model architecture was provided!')
            logging.warning('Either provide a valid torchvision architecture by using --arch "[architecturename]",')
            logging.warning('or load an existing checkpoint!')
    
    def build_classifier(self):
        
        logging.info('# Building new classifier with {} hidden units'.format(self.hidden_units))

        # get old in_features form model to get the input size of the new classifier
        self.input_size = self.model.classifier[0].in_features

        # freeze params of pretrained network
        for param in self.model.parameters():
            param.requires_grad = False

        # init input layer 
        modules = list()
        modules.append(nn.Linear(self.input_size,self.hidden_units[0]))
    
        # add hidden layers and output layer
        for index, size in self.hidden_units:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.2))
            output = self.hidden_units[index+1] if index < (len(self.hidden_units)-1) else self.output_size
            modules.append(nn.Linear(size, output))

        # add activation function
        modules.append(nn.LogSoftmax(dim=1))
                    
        # build ans set classifier by unpacking modules into a nn.Sequential 
        self.model.classifier = nn.Sequential(*modules)   

        # info
        logging.info('# Setting new classifier to:')
        logging.info(self.model.classfier)  
    
    def train_model(self, train_dataloader, validation_dataloader):
        
        logging.info('# Training Started!')

        # build new classifier for training
        self.model.classifier = self.build_classifier()

         # define criterion and optimizer 
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(),lr=self.learning_rate)

        # move model to gpu
        logging.info('Moving model to device: {}'.format(self.device))
        self.model.to(self.device)
        
        steps = 0

        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in train_dataloader:
                steps +=1
                # Move input and label tensors to the GPU
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                log_ps = self.model(images)
                train_loss = criterion(log_ps, labels)
                train_loss.backward()
                optimizer.step()
                running_loss += train_loss.item()
                
                if steps % self.print_info_every == 0:
                    validation_loss, validation_accuracy = self.calc_loss_accuracy(validation_dataloader, criterion)
                    no_items = len(validation_dataloader)
                    print(
                        "epoch: {}/{}".format(epoch+1,self.epochs),
                        "Train loss: {:.3f}..".format(running_loss/self.print_info_every),
                        "Validation loss: {:.3f}..".format(validation_loss/no_items),
                        "Validation Accuracy: {:.3f}".format(validation_accuracy/no_items)
                    )             
                    running_loss=0
                    
    def calc_loss_accuracy(self, dataloader):
        loss = 0
        accuracy = 0 
        criterion = nn.NLLLoss()
        
        # go into eval mode 
        self.model.eval()  

        with torch.no_grad():
            # validation pass here
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                log_ps = self.model(images)
                loss += criterion(log_ps,labels).item()
                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # go back to train mode 
        self.model.train()

        return loss, accuracy

    def test_model(self, dataloader):
        no_items = len(dataloader)

        # calc loss and accuracy 
        test_loss, test_accuracy = self.calc_loss_accuracy(dataloader)
        
        # print results 
        print(
            "Test loss: {:.3f}..".format(test_loss/no_items),
            "Test Accuracy: {:.3f}".format(test_accuracy/no_items)
        )             

    def save_checkpoint(self, checkpoint_name, class_to_idx): 
        # create checkpoint
        checkpoint = {
            'input_size': self.input_size,
            'hidden_units': self.hidden_units,
            'output_size': self.output_size, 
            'class_to_idx': class_to_idx,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        # save checkpoint
        torch.save(checkpoint, checkpoint_name)                    


    def load_checkpoint(self, model, filepath):
        # get checkpoint
        checkpoint = torch.load(filepath)
        
        logging.info('# loading checkpoint')

        for hyperparam in ['input_size', 'hidden_units', 'output_size', 'epochs', 'architecture']:
            setattr(self, hyperparam, checkpoint[hyperparam])
            logging.info(hyperparam+': {}'.format(getattr(self, hyperparam)))
        
        # get model according to architecture
        try:
            self.model = TorchvisionModels(self.architecture).load()
        except UnknownModelError as error:
            logging.error(error)

        # build and set classifier such that the loaded checkpoint can be applied correctly
        self.model.classifier = self.build_classifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        
        # set and load optimizer
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model = model 
        self.optimizer = optimizer

# utility class to load pretrained models from torchvision.models 
class TorchvisionModels: 
    # constructor
    def __init__(self, architecture):
        self.architecture = architecture

    def load(self): 
        if hasattr(models, self.architecture):
            return models.getattr(self.architecture)(pretrained=True) 
        else: 
            raise UnknownModelError('the entered model is not part of torchvision.models') 

class UnknownModelError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message