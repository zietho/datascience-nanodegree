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
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument("--tok_k", 
                    dest="top_k", 
                    default="5",
                    action="store",
                    help="sets number of top results")
parser.add_argument("--category_names", 
                    dest="category_names", 
                    action="store",
                    help="sets the category names for the results")
parser.add_argument("--gpu", 
                    dest="device",
                    action="store_const",
                    const="cuda",
                    default="cpu",  
                    help="flag for deciding whether gpu is used or not")

# retrieve all given CLI arguments
args = parser.parse_args()

# image processing 
image_utilities = ImageUtilities({
    'image_size': args.image_size,
    'batch_size': args.batch_size, 
    'mean': args.mean,
    'std': args.std
})

image_path = 'flowers/test/2/image_05107.jpg'
image = Image.open(image_path)
processed_image = process_image(image)
imshow(processed_image)