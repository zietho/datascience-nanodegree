# IMPORTS
from argparse import ArgumentParser # for parsing all command line arguments
from network import Network
from network import TorchvisionModels
from network import UnknownModelError
from utilities import ImageUtilities
from torch import optim
from torch import nn

# define argument parser 
parser = ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument("--save_dir", 
                    dest="save_dir", 
                    help="sets the directory where checkpoints are stored")
parser.add_argument("--arch", 
                    dest="architecture", 
                    default="vgg19",
                    action="store",
                    help="sets the architecture that is used for the newtork")
parser.add_argument("--epochs", 
                    dest="epochs", 
                    default=5,
                    help="sets the number of epochs used during training")
parser.add_argument("--learning_rate", 
                    dest="learning_rate",
                    default=0.001,  
                    help="sets the architecture that is used for the newtork")
parser.add_argument("--gpu", 
                    dest="device",
                    action="store_const",
                    const="cuda",
                    default="cpu",  
                    help="flag for deciding whether gpu is used or not")
parser.add_argument("--mean", 
                    dest="mean",  
                    default=[0.485, 0.456, 0.406],
                    help="set the mean of the data set used")
parser.add_argument("--std", 
                    dest="std",
                    std = [0.229, 0.224, 0.225],
                    help="set the std of the data set used")
parser.add_argument("--image_size", 
                    dest="std",  
                    default=224,
                    help="set image size")
parser.add_argument("--batch_size", 
                    dest="std",  
                    default=32,
                    help="set batch size used for training")
parser.add_argument("--hidden_units", 
                    dest="no_hidden_units",  
                    default=3,
                    help="set batch size used for training")

# retrieve all given CLI arguments
args = parser.parse_args()

# Load pretrained model from torchvision models based on given architecture
try:
    model = TorchvisionModels(args.architecture).load()
except UnknownModelError as err:
    print('An error occured ', err)

# create network based on arguments dict 
network = Network(model, {
    'data_dir': args.data_dir,
    'save_dir': args.save_dir,
    'epochs' : args.epochs,
    'learning_rate': args.learning_rate,
    'device': args.gpu,
    'hidden_layers': args.hidden_layers,
})

# image processing 
image_utilities = ImageUtilities({
    'image_size': args.image_size,
    'batch_size': args.batch_size, 
    'mean': args.mean,
    'std': args.std
})

# load data from using the utilities function
image_datasets, data_loaders = image_utilities.load(args.data_dir)

#train only the classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
train_dataloader = data_loaders.get('train')
validation_dataloader = data_loaders.get('valid')  
test_dataloader = data_loaders.get('test')

network.train_model(train_dataloader, validation_dataloader, criterion, optimizer)

network.validate_model(test_dataloader, criterion)
# set class to idx 
network.set_class_to_idx(image_datasets.get('train').class_to_idx)


#test arguments
#args = parser.parse_args()
#print(args)

# check
image_path = 'flowers/test/2/image_05107.jpg'
image = Image.open(image_path)
processed_image = process_image(image)
imshow(processed_image)