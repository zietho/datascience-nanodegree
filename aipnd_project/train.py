# IMPORTS
from argparse import ArgumentParser # for parsing all command line arguments
from network import Network, UnknownModelError, UnsupportedArchitectureError
from utilities import ImageUtilities
import logging
import sys

__author__ = "Thomas Ziegelbecker"
__name__ = 'main'

# fn to get all possible cli arguments 
def get_cli_arguments():
    parser = ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument("--save_dir", 
                        dest="save_dir", 
                        help="sets the directory where checkpoints are stored")
    parser.add_argument("--arch", 
                        dest="architecture", 
                        default="vgg16",
                        action="store",
                        help="sets the architecture that is used for the newtork")
    parser.add_argument("--epochs", 
                        dest="epochs", 
                        default=1,
                        help="sets the number of epochs used during training")
    parser.add_argument("--learning_rate", 
                        dest="learning_rate",
                        default=0.001,  
                        help="sets the learning rate used during training")
    parser.add_argument("--hidden_units", 
                        dest="hidden_units",  
                        default=0,
                        help="set the number of hidden units in the new classifier")
    parser.add_argument("--gpu", 
                        dest="device",
                        action="store_const",
                        const="cuda",
                        default="cpu",  
                        help="flag for deciding whether gpu is used or not")
    # for convenience INFO contains all different infos between the steps
    parser.add_argument("--logging", 
                        dest="logging_level",  
                        default='WARNING',
                        help="set the number of hidden units in the new classifier")
    
    return parser.parse_args()

def main(): 
    args = get_cli_arguments()
   
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.logging_level)) 

    # log all provided arguments
    logging.info('Arguments used:')
    for key,value in vars(args).items():
        logging.info(str(key)+':'+str(value))

    # instantiate network class based on provided arguments 
    try:
        network = Network({
            'epochs' : args.epochs,
            'learning_rate': args.learning_rate,
            'hidden_units': args.hidden_units,
            'device': args.device,
            'logging_level': args.logging_level,
            'architecture': args.architecture
        })  

        # instantiate image utilities class to load image data  
        image_utilities = ImageUtilities({
            'image_size': 224,
            'batch_size': 12, 
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'logging_level': args.logging_level
        })

        # load image datasets and train loaders
        image_datasets, data_loaders = image_utilities.load(args.data_dir)
        
        # train model
        network.train_model(data_loaders.get('train'), data_loaders.get('valid'))

        # test model 
        network.test_model(data_loaders.get('test'))

        # save model
        network.save_checkpoint(args.save_dir, image_datasets.get('train').class_to_idx)
    except UnsupportedArchitectureError as e:
        logging.error(e.message)
    except UnknownModelError as e:
        logging.error(e.message)

    

    

main()