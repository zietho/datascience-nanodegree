import logging
import json
import numpy as np
import time
import sys
import torch
import os
import datetime
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import models

class Network: 
    #constructor 
    def __init__(self, args): 
        # init setable hyperparameters
        self.epochs = int(args.get('epochs')) if 'epochs' in args.keys() else 5
        self.learning_rate = float(args.get('learning_rate')) if 'learning_rate' in args.keys() else 0.001
        self.device = str(args.get('device')) if 'device' in args.keys() else 'cpu'
        self.architecture = str(args.get('architecture')) if 'architecture' in args.keys() else None
        self.hidden_units = int(args.get('hidden_units')) if 'hidden_units' in args.keys() else 512
        self.top_k = int(args.get('top_k')) if 'top_k' in args.keys() else 3
        self.category_names_file = args.get('category_names') if 'category_names' in args.keys() else 'cat_to_name.json'
        
        # set log level
        self.logging_level = str(args.get('logging_level')) if 'logging_level' in args.keys() else getattr(logging, 'WARNING')
        logging.basicConfig(stream=sys.stderr, level=self.logging_level)
        
        if self.architecture != None:
            # architecture check
            self.supported_architectures = ['densenet', 'resnet', 'vgg']
            architecture_supported = False 
            for supported_architecture in self.supported_architectures:
                if self.architecture.find(supported_architecture) != -1:
                    architecture_supported = True

            if architecture_supported == False:
                error_message = 'The given architecture is not supported! Pls use one of the following: '
                supported = ''.join(map(lambda x: x+', ',self.supported_architectures))
                raise UnsupportedArchitectureError(error_message+supported)

            # fetch selected model by architecture
            self.model = TorchvisionModels(self.architecture).load()
        
            # init optimizer and output size
            self.optimizer = None
            self.output_size = 102 

        # set category names files
        if not os.path.exists(self.category_names_file):
            raise NoCategoryNamesFileError('Provided files {} does not exist!'.format(self.category_names_file))
        else: 
            with open(self.category_names_file, 'r') as file:
                self.category_names = json.load(file)

    def __get_classifier_by_model(self):
        if self.architecture.find('vgg') != -1:
            return self.model.classifier
        elif self.architecture.find('densenet') != -1:
            return self.model.classifier
        elif self.architecture.find('resnet') != -1:
            return self.model.fc
        else:   
            logging.error('#architecture not supported!')
            return 0

    def __set_classifier_by_model(self, new_classifier):

        # info
        logging.info('# Setting new classifier to:')
        logging.info(new_classifier)
        if self.architecture.find('vgg') != -1:
            self.model.classifier = new_classifier
        elif self.architecture.find('densenet') != -1:
            self.model.classifier = new_classifier
        elif self.architecture.find('resnet') != -1:
            self.model.fc = new_classifier
        else:   
            logging.error('#architecture not supported!')

    def __get_input_size_by_model(self):
        classifier = self.__get_classifier_by_model()
        if self.architecture.find('vgg') != -1:
            return classifier[0].in_features
        else:
            return classifier.in_features
    
    def build_classifier(self):
        logging.info('# Building new classifier with {} hidden units'.format(self.hidden_units))

        self.input_size = self.__get_input_size_by_model()         
        
        # freeze params of pretrained network
        for param in self.model.parameters():
            param.requires_grad = False

        multiplier = (self.input_size - self.output_size) // (self.hidden_units+1)
        logging.info('# input nodes: '+str(self.input_size))
        logging.info('# hidden units: '+str(self.hidden_units))
        logging.info('# output nodes: '+str(self.output_size))
        # init input layer 
        modules = list()
        modules.append(nn.Linear(self.input_size,self.input_size - multiplier))
        
        # add hidden layers and output layer
        input_nodes, output_nodes = 0,0
        hidden_layers = range(1,self.hidden_units+1)
        
        for i, val in enumerate(hidden_layers):
            input_nodes = self.input_size - (multiplier * val)
            if i < (len(hidden_layers)-1):
                output_nodes = self.input_size - (multiplier * hidden_layers[i+1])
            else:
                output_nodes = self.output_size
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.2))
            modules.append(nn.Linear(input_nodes, output_nodes))

        # add activation function
        modules.append(nn.LogSoftmax(dim=1))
                    
        # build and set classifier by unpacking modules into a nn.Sequential 
        self.__set_classifier_by_model(nn.Sequential(*modules))   
    
    
    def train_model(self, train_dataloader, validation_dataloader):
        logging.info('# Training Started!')

        # build new classifier 
        self.build_classifier()

        # move model to device
        self.model.to(self.device)
        logging.info('# model moved to {}'.format(self.device)) 
        
        # define criterion and optimizer
        criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.__get_classifier_by_model().parameters(),lr=self.learning_rate)
        steps = 0
        print_info_every = 50
        no_items = len(validation_dataloader)

        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in train_dataloader:
                steps +=1
                # Move input and label tensors to the GPU
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                log_ps = self.model(images)
                train_loss = criterion(log_ps, labels)
                train_loss.backward()
                self.optimizer.step()
                running_loss += train_loss.item()
                
                if steps % print_info_every == 0:
                    validation_loss, validation_accuracy = self.calc_loss_accuracy(validation_dataloader, criterion)
                    print(
                        "epoch: {}/{}".format(epoch+1,self.epochs),
                        "train loss: {:.3f}..".format(running_loss/print_info_every),
                        "validation loss: {:.3f}..".format(validation_loss/no_items),
                        "validation accuracy: {:.3f}".format(validation_accuracy/no_items)
                    )             
                    running_loss=0
                    
    def calc_loss_accuracy(self, dataloader, criterion=nn.NLLLoss()):
        loss = 0
        accuracy = 0 
        
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

    def save_checkpoint(self, save_dir, class_to_idx): 
        # create saving path and checkpoint
        
        if save_dir != None and not os.path.isdir(save_dir):
            logging.info('# save directory does not exit, thurs creating: {}'.format(save_dir))
            os.makedirs(save_dir)
        checkpoint_name = 'checkpoint-'+str(datetime.datetime.today().strftime('%Y-%m-%d'))+'.pth.tar'    
        checkpoint_path = ''
        if save_dir != None:
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
        else:
            checkpoint_path = checkpoint_name
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
        logging.info('# saving checkpoint to {}'.format(checkpoint_path))
        torch.save(checkpoint, checkpoint_path)                    


    def load_checkpoint(self, checkpoint_path):
        logging.info('# loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        
        for hyperparam in ['input_size', 'hidden_units', 'output_size', 'epochs', 'architecture']:
            setattr(self, hyperparam, checkpoint[hyperparam])
            logging.info(hyperparam+': {}'.format(getattr(self, hyperparam)))
        
        # get model according to architecture
        try:
            self.model = TorchvisionModels(self.architecture).load()
        except UnknownModelError as error:
            logging.error(error)

        # build and set classifier such that the loaded checkpoint can be applied correctly
        self.build_classifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        
        # set and load optimizer
        self.optimizer = optim.Adam(self.__get_classifier_by_model().parameters(), lr=self.learning_rate) 
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def predict(self, image):
        if self.model is None:
            raise NoModelProvidedError('No model provided! E.g., load checkpoint first!')
        else:
            self.model.eval()
        
        img = torch.from_numpy(image).float()
        img = img.unsqueeze(0)

        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = self.model.forward(img)
            ps = torch.exp(output)
            probs, classes =  ps.data.topk(self.top_k)
            # Map classes to indices 
            inverted_class_to_idx = {self.model.class_to_idx[k]: k for k in self.model.class_to_idx}
            
            mapped_classes = list()
            for label in classes.numpy()[0]:
                mapped_classes.append(inverted_class_to_idx[label])
            
            # map the inverted classes now to the given category names
            categories = [self.category_names.get(x) for x in mapped_classes]
        
        # Return results
        return probs.numpy()[0], categories

# utility class to load pretrained models from torchvision.models 
class TorchvisionModels: 
    # constructor
    def __init__(self, architecture):
        self.architecture = architecture

    def load(self): 
        if hasattr(models, self.architecture):
            return getattr(models,self.architecture)(pretrained=True) 
        else: 
            raise UnknownModelError('the entered model is not part of supported torchvision.models') 

class UnknownModelError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message

class NoModelProvidedError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message

class NoCategoryNamesFileError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message

class UnsupportedArchitectureError(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message
    def __str__(self):
        return self.message