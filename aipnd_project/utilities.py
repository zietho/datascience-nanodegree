import torch
from torchvision import datasets, transforms
from PIL import Image

class ImageUtilities:
    #constructor 
    def __init__(self, args):
        self.data_dir = args.data_dir if hasattr(args,'data_dir') else '/'
        self.mean = args.mean if hasattr(args,'mean') else [0.485, 0.456, 0.406],
        self.std = args.std if hasattr(args,'std') else [0.229, 0.224, 0.225],
        self.batch_size = args.batch_size if hasattr(args,'batch_size') else 32
        self.image_size = args.image_size if hasattr(args,'image_size') else 224

    def load_data(self):
        # set up directories for the three datasets
        folders = {key: self.data_dir+'/{}'.format(key) for key in ['train', 'valid', 'test']}

        # define transforms
        self.data_transforms = {
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
        self.image_datasets = {
            key: datasets.ImageFolder(folders.get(key),transform=val) for key,val in self.data_transforms.items()
            }
                
        self.class_to_idx = { 
            key: value.class_to_idx for key,value in self.image_datasets.items()
            }
        
        # Using the image datasets and the trainforms, define the dataloaders
        self.data_loaders = {
            'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'valid': torch.utils.data.DataLoader(self.image_datasets['valid'], batch_size=self.batch_size),
            'test': torch.utils.data.DataLoader(self.image_datasets['test'], batch_size=self.batch_size)
            } 

    def get_data_loaders(self):
        if len(self.data_loaders)>0:
            return self.data_loaders
        else:
            return None
    
    def get_class_to_idx(self):
        if len(self.get_class_to_idx)>0:
            return self.get_class_to_idx
        else:
            return None
    
    