# IMPORTS
from argparse import ArgumentParser # for parsing all command line arguments
from network import Network
from utilities import ImageUtilities
import logging
import sys

__author__ = "Thomas Ziegelbecker"
__name__ == "__main__"

# fn to get all possible cli arguments 
def get_cli_arguments():
    parser = ArgumentParser()
    parser.add_argument('image',
                        help="provide the path to an image")
    parser.add_argument('checkpoint',
                        help="provide the path to an existing checkpoint")
    parser.add_argument("--top_k", 
                        dest="top_k",
                        action="store",
                        default="3",
                        help="sets number of top results")
    parser.add_argument("--category_names", 
                        dest="category_names", 
                        action="store",
                        default="cat_to_name.json",
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

    return parser.parse_args()

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

     # instantiate network class based on provided arguments 
    network = Network({
        'device': args.device,
        'top_k': args.top_k,
        'category_names': args.category_names,
        'logging_level': args.logging_level
    })

    # load given checkpoint ans by that set necessary model 
    network.load_checkpoint(args.checkpoint)

    # get and process given image  
    image = image_utilities.open(args.image)
    processed_image = image_utilities.process(image, 256, 224)
    
    # predict
    probs, classes = network.predict(processed_image)

    # print results
    for i, val in enumerate(classes):
        print(str(i+1)+' - '+val+' : '+str(probs[i]))

main()