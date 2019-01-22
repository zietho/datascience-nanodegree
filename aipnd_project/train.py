# IMPORTS
from argparse import ArgumentParser # for parsing all command line arguments
from network import Network
from utilities import ImageUtilities
from torch import optim
from torch import nn

# start main 
main()

def main(): 

    args = get_cli_arguments()

    # instantiate network class based on provided arguments 
    network = Network({
        'epochs' : args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_layers': args.hidden_layers,
        'device': args.device,
        'logging_level': args.logging_level
    })

    # instantiate image utilities class to load image data  
    image_utilities = ImageUtilities({
        'image_size': args.image_size,
        'batch_size': args.batch_size, 
        'mean': args.mean,
        'std': args.std
    })

    # load image datasets 
    image_datasets, data_loaders = image_utilities.load(args.data_dir)

    # get dataloaders
    train_dataloader = data_loaders.get('train')
    validation_dataloader = data_loaders.get('valid')  
    test_dataloader = data_loaders.get('test')
    
    # train model
    network.train_model(train_dataloader, validation_dataloader)

    # test model 
    network.test_model(test_dataloader)

    # save model
    network.save_checkpoint(args.save_dir, image_datasets.get('train').class_to_idx)


# fn to get all possible cli arguments 
def get_cli_arguments():
    
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
                        help="sets the learning rate used during training")
    parser.add_argument("--hidden_units", 
                        dest="hidden_units",  
                        default=3,
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



