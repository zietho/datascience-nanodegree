import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageUtilities:
    #constructor 
    def __init__(self, args):
        self.mean = args.mean if hasattr(args,'mean') else [0.485, 0.456, 0.406],
        self.std = args.std if hasattr(args,'std') else [0.229, 0.224, 0.225],
        self.batch_size = args.batch_size if hasattr(args,'batch_size') else 32
        self.image_size = args.image_size if hasattr(args,'image_size') else 224

    def load(self, data_dir):
        # set up directories for the three datasets
        folders = {key: data_dir+'/{}'.format(key) for key in ['train', 'valid', 'test']}

        # define transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean,self.std)
                ]),
            'valid': transforms.Compose([
                transforms.Resize(size=[self.image_size,self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean,self.std)]),
            'test': transforms.Compose([
                transforms.Resize(size=[self.image_size,self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean,self.std)
                ])
            } 
            
        # Load the datasets with ImageFolder
        image_datasets = {
            key: datasets.ImageFolder(folders.get(key),transform=val) for key,val in data_transforms.items()
            }
                
        class_to_idx = { 
            key: value.class_to_idx for key,value in image_datasets.items()
            }
        
        # Using the image datasets and the trainforms, define the dataloaders
        data_loaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=self.batch_size),
            'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=self.batch_size)
            } 

        return image_datasets, data_loaders
    
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
    
    

    def imshow(self, image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array(self.mean)
        std = np.array(self.std)
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax
        
    
    

    