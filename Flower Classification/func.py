import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib import pyplot as plt
import json
# I define this up here so I can use it for image processing to make predictions
test_val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def get_names_and_probabilities(cat_names_path,probs,classes,cti):
    if cat_names_path is None:
        return probs,classes
    with open(cat_names_path, 'r') as f:
        cat_to_name = json.load(f)
    index_to_class = {value:key for key,value in zip(cti.keys(),cti.values())}
    index_to_name = {key:cat_to_name[value] for key,value in zip(index_to_class.keys(),index_to_class.values()) }
    names = [index_to_name[flower_index]for flower_index in classes.tolist()[0] ]
    return probs,names 
def load_data(path):
    """
    Returns: [trainloader,testloader,valloader,class_to_idx]
    """
    train_dir = path + "/train"
    test_dir = path + "/test"
    valid_dir = path + "/valid"
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_val_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=test_val_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)

    return trainloader,testloader,valloader,train_data.class_to_idx

def get_network_and_extras(modelname,  hidden_units, learning_rate,):
    '''
    Gets hyperparameters and builds a network. 
    Returns: [model, optimizer, loss_func]
    '''
    if modelname == "vgg" or modelname == "vgg16" or modelname == None:
        model = models.vgg16(pretrained = True)
        classifier = nn.Sequential(nn.Linear(25088,hidden_units),nn.ReLU(),nn.Dropout(p = 0.2),
                           nn.Linear(hidden_units,256),nn.ReLU(),nn.Dropout(p = 0.2),
                           nn.Linear(256  ,102),nn.LogSoftmax(dim = 1))    
    elif modelname == "alexnet":
        model = models.alexnet(pretrained = True)
        classifier = nn.Sequential(nn.Linear(9216,hidden_units),nn.ReLU(),nn.Dropout(p = 0.2),
                           nn.Linear(hidden_units  ,256),nn.ReLU(),nn.Dropout(p = 0.2),
                           nn.Linear(256  ,102),nn.LogSoftmax(dim = 1))
    else:
        raise ValueError("Incompatible architecture!")
    for parameter in model.parameters():
            parameter.requires_grad = False
    model.classifier = classifier
    optimizer = torch.optim.Adam(params = model.classifier.parameters(),lr = learning_rate)
    loss_func = nn.NLLLoss()
    return [model, optimizer, loss_func]
def savemodel(model,savepath,class_to_idx,epoch,optimizer,learning_rate,loss_func,modelname,hidden_units):
    checkpoint = {
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict":optimizer.state_dict(),
                  "loss_func_state_dict":loss_func.state_dict(),
                  "current_epoch":epoch,
                  "learning_rate":learning_rate,
                  "class_to_idx_dict":class_to_idx,
                  "model_name":modelname,
                  "hidden_units":hidden_units
                }
    if savepath is None:
        torch.save(checkpoint,"checkpoint.pth")
    elif savepath[len(savepath)-1] !="/":
        torch.save(checkpoint, savepath + "/checkpoint.pth")
    else:
        torch.save(checkpoint, savepath + "checkpoint.pth")
    print("Succesfully saved model as checkpoint.pth!")
def train_validate_and_save_model(model,optimizer, loss_func,epochs,trainloader,valloader,gpu,savepath,class_to_idx,learning_rate,modelname,hidden_units):
    #training the model

    if gpu is not None:
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    val_loss_func = nn.NLLLoss()
    batch = 0
    total_train_accuracy = 0
    total_val_accuracy = 0
    total_train_loss = 0
    total_val_loss = 0
    val_batch = 0
    print_every = 5
    model.train()
    for epoch in range(epochs):
        print(f"======= START OF EPOCH {epoch+1}/{epochs} =======")
        #Train batch
        for images,labels in trainloader:
            batch += 1
            #Stepping the optimizer
            labels,images = labels.to(device),images.to(device)
            optimizer.zero_grad()
            log_pred = model.forward(images)
            loss = loss_func(log_pred,labels)
            loss.backward()
            pred = torch.exp(log_pred).topk(1,dim = 1)[1]
            optimizer.step()
            
            #Train Accuracy
            labels.resize_(pred.shape)
            equality = pred == labels
            total_train_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            total_train_loss += loss.item()
            if not(batch%print_every):
                print("Train Accuracy:{:.3f}% ".format(total_train_accuracy / batch*100),
                      "Train Loss: {:.3f}".format(total_train_loss/batch))
        #Validation Batch
        for val_images,val_labels in valloader:
            val_batch +=1
            #Getting predictions
            val_images,val_labels = val_images.to(device), val_labels.to(device)
            val_log_pred = model.forward(val_images)
            val_pred = torch.exp(val_log_pred).topk(1, dim = 1)[1]
            val_loss = val_loss_func(val_log_pred,val_labels)
            val_labels.resize_(val_pred.shape)
            total_val_loss += val_loss.item()
            #Validation accuracy
            val_equality = val_pred == val_labels
            total_val_accuracy += torch.mean(val_equality.type(torch.FloatTensor))
            if not(val_batch % print_every):
                print("Validation Accuracy:{:.3f}%".format(total_val_accuracy / val_batch*100),"Validation Loss:{:.3f}".format(total_val_loss / val_batch))
    print("Succesfully trained model! Saving the model...")
    savemodel(model,savepath,class_to_idx,epoch,optimizer,learning_rate,loss_func,modelname,hidden_units)
def load_checkpoint_and_make_predictions(checkpoint_path,image_path,topk,cat_names_path,gpu):
    checkpoint = torch.load(checkpoint_path)
    if "model_name" in checkpoint.keys():
        model_name = checkpoint["model_name"]
    else:
        model_name = "vgg16"
    if "hidden_units" in checkpoint.keys():
        hidden_units = checkpoint["hidden_units"]
    else:
        hidden_units = 512
    if model_name == "vgg" or model_name == "vgg16":
        model = models.vgg16(pretrained = True)
        input_size = 25088
    if model_name =="alexnet" or model_name=="Alexnet":
        model = models.alexnet(pretrained = True)
        input_size = 9216
    model.class_to_idx = checkpoint["class_to_idx_dict"]
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = nn.Sequential(nn.Linear(input_size,hidden_units),nn.ReLU(),nn.Dropout(p = 0.2),
                               nn.Linear(hidden_units  ,256),nn.ReLU(),nn.Dropout(p = 0.2),
                               nn.Linear(256  ,102),nn.LogSoftmax(dim = 1))
    model.eval()
    model.classifier = classifier
    model.load_state_dict(checkpoint["model_state_dict"])
    probs,classes = predict(model,image_path,topk,gpu)
    return get_names_and_probabilities(cat_names_path,probs,classes,model.class_to_idx)
def process_image(image):
    return test_val_transforms(image)
def predict(model,image_path,topk,gpu):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = "cpu"
    if gpu is not None:
        device = "cuda"
    model = model.to(device)
    image = process_image(Image.open(image_path))
    image.resize_(1,3,224,224)
    image = image.to(device)
    log_pred = model.forward(image)
    pred = torch.exp(log_pred)
    probs,classes = pred.topk(topk,dim = 1)
    return probs,classes