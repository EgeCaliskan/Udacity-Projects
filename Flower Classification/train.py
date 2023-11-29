import argparse
import func 
parser = argparse.ArgumentParser(description='Train some machine learning algorithms!.')
parser.add_argument('data_dir', type=str,
                    help='The file path for the data directory')
parser.add_argument('--save_dir',dest= "save_dir",type = str,
                    help='The file path for the checkpoint save directory')
parser.add_argument('--learning_rate',type = str,
                    help='Specify the learning rate at which the algorithm learns.')
parser.add_argument('--hidden_units',type = int,
                    help='The amount of hidden units.')
parser.add_argument('--epochs',type = int,
                    help='Specify how many epochs the model should train for')
parser.add_argument('--gpu',action = 'store_true',
                    help='Use GPU for training')
parser.add_argument('--arch',type = str,
                    help='Specify the architecture to train the model on')
args = parser.parse_args()


data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = 0.002 if args.learning_rate is None else args.learning_rate
hidden_units = 512 if args.hidden_units is None else args.hidden_units
epochs = 10 if args.epochs is None else args.epochs
gpu = args.gpu
arch = "vgg16" if args.arch is None else args.arch

model,optimizer,loss_func = func.get_network_and_extras(arch,hidden_units,learning_rate)
trainloader,testloader,valloader,class_to_idx = func.load_data(data_dir)
func.train_validate_and_save_model(model,optimizer,loss_func,epochs,trainloader,valloader,gpu,save_dir,class_to_idx,learning_rate,arch,hidden_units)
