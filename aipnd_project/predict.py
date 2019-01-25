# IMPORTS
from argparse import ArgumentParser # for parsing all command line arguments
from network import Network
from network import TorchvisionModels
from network import UnknownModelError
from utilities import ImageUtilities
import logging
import sys

__author__ = "Thomas Ziegelbecker"
__name__ == "__main__"

# fn to get all possible cli arguments 
def get_cli_arguments():
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
    parser.add_argument("--logging", 
                        dest="logging_level",
                        action="store",
                        default="WARNING",
                        help="flag for deciding whether gpu is used or not")

    # retrieve all given CLI arguments
    return parser.parse_args()



#äimage_path = 'flowers/test/2/image_05107.jpg'
#image = Image.open(image_path)
#processed_image = process_image(image)
#imshow(processed_image)

def main():
    args = get_cli_arguments()
    # set log level
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.logging_level)) #getattr(logging,args.logging_level))
    
    # log all provided arguments
    for key,value in vars(args).items():
        logging.info(str(key)+':'+str(value))

    # instantiate image utilities class to load image data  
    image_utilities = ImageUtilities({
        'image_size': 224,
        'batch_size': 32, 
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'logging_level': args.logging_level
    })

    
        
main()