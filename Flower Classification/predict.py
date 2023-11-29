import argparse
import func
import torch
parser = argparse.ArgumentParser(description='Use a trained machine learning algorithm to make predictions!')
parser.add_argument('checkpoint',type = str,
                    help='Specify the location of the checkpoint to load the model')
parser.add_argument('image_path',type =str,
                    help = 'Specify the location of the image to predict')
parser.add_argument('--topk',type = int,
                    help='Get the top k most likely classes')
parser.add_argument('--category_names',
                    help='Map from categories to names')
parser.add_argument('--gpu',action = 'store_true',
                    help = 'Use gpu for inference')
args = parser.parse_args()
gpu = args.gpu
checkpoint = args.checkpoint 
cat_names_path = args.category_names
topk = 1 if args.topk is None else args.topk
image_path = args.image_path
probs,classes = func.load_checkpoint_and_make_predictions(checkpoint,image_path,topk,cat_names_path,gpu)
classes = classes.tolist()[0] if type(classes) == torch.Tensor else classes
probs =  probs.tolist()[0]
print("Class Probabilities:", {c:p for c,p in zip(classes,probs)})