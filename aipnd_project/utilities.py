from PIL import Image
from torchvision import datasets, transforms
import logging
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class ImageUtilities:
    #constructor 
    def __init__(self, args):
        self.mean = args.get('mean') if 'mean' in args.keys() else [0.485, 0.456, 0.406]
        self.std = args.get('std') if 'std' in args.keys() else [0.229, 0.224, 0.225]
        self.batch_size = args.get('batch_size') if 'batch_size' in args.keys() else 32
        self.image_size = args.get('image_size') if 'image_size' in args.keys() else 224
        self.logging_level = args.get('logging_level') if 'logging_level' in args.keys() else getattr(logging, 'WARNING')
         # set log level
        logging.basicConfig(stream=sys.stderr, level=self.logging_level)

    def load(self, data_dir):
        # set up directories for the three datasets
        folders = {key: data_dir+'/{}'.format(key) for key in ['train', 'valid', 'test']}
        logging.info(self.mean)
        logging.info(self.std)
        # define transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,std=self.std)
                ]),
            'valid': transforms.Compose([
                transforms.Resize(size=[self.image_size,self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,std=self.std)]),
            'test': transforms.Compose([
                transforms.Resize(size=[self.image_size,self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,std=self.std)
                ])
            } 
            
        # Load the datasets with ImageFolder
        image_datasets = {
            key: datasets.ImageFolder(folders.get(key),transform=val) for key,val in data_transforms.items()
            }
        
        # Using the image datasets and the trainforms, define the dataloaders
        data_loaders = {
            'train': torch.utils.data.DataLoader(image_datasets.get('train'), batch_size=self.batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(image_datasets.get('valid'), batch_size=self.batch_size),
            'test': torch.utils.data.DataLoader(image_datasets.get('test'), batch_size=self.batch_size)
            } 
        
        # show
        logging.info(data_transforms)
        logging.info(image_datasets)
        logging.info(data_loaders)

        return image_datasets, data_loaders

    def open(self, image_path):
        if not os.path.exists(image_path):
            logging.info('# no image found with the given path {}'.format(image_path))
            sys.exit(0)
        return Image.open(image_path)
        
    def process(self, image, new_size, crop_size):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # copy image, convert to rgb and get old width and height 
        pil_image = image.copy()
        pil_image = pil_image.convert('RGB')
        old_width, old_height = pil_image.size
        
        # calc new width and height
        if old_width < old_height:
            new_height=int(new_size*(old_width/old_height))
            new_width=new_size
        else:
            new_width=int(new_size*(old_width/old_height))
            new_height=new_size
        
        # scale
        pil_image = pil_image.resize([new_width,new_height])
        
        # crop    
        width, height = pil_image.size   # Get dimensions
        left = (width - crop_size)/2
        top = (height - crop_size)/2
        right = (width + crop_size)/2
        bottom = (height + crop_size)/2
        
        pil_image = pil_image.crop((left, top, right, bottom))
        
        # normalize 
        # - convert to numpy array
        # - normalize by subtracting the mean from all color channels and dividing by the std deviation
        mean = np.array(self.mean)
        std = np.array(self.std)
        numpy_image = np.array(pil_image) / 255
        numpy_image_normalized = (numpy_image - mean) / std
        
        # transpose
        transposed = np.transpose(numpy_image_normalized,(2,0,1))
        
        # return tensor
        return transposed

