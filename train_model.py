#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile


try:
    import smdebug.pytorch as smd
    from smdebug import modes
except:
    pass



#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.eval()
    
    try:
        hook.set_mode(modes.EVAL)
    except:
        pass

    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()

    total_acc = 100 * running_corrects / len(test_loader.dataset)

    print("Test set: Accuracy: {:.0f}%\n".format(total_acc))
    
    return total_acc

def train(model, train_loader, criterion, optimizer, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    
    model.train()

    try:
        hook.set_mode(modes.TRAIN)
    except:
        pass

    running_loss=0
    correct=0
    for data, target in train_loader:
        optimizer.zero_grad()
        #NOTE: Notice how we are not changing the data shape here
        # This is because CNNs expects a 3 dimensional input
        pred = model(data)
        loss = criterion(pred, target)
        running_loss+=loss
        loss.backward()
        optimizer.step()
        pred=pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    print("\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            running_loss/len(train_loader.dataset), correct, len(train_loader.dataset), 100.0 * correct / len(train_loader.dataset)))
    

    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    feat_count = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(nn.Linear(feat_count, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
            shuffle=True)
    
    return data_loader

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    try:
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
    except:
        pass

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    train_data = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'dogImages/train/'), transform=transform)
    val_data = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'dogImages/valid/'), transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'dogImages/test/'), transform=transform)
    
    train_loader = create_data_loaders(train_data, args.batch_size)
    val_loader = create_data_loaders(val_data, args.batch_size)
    test_loader = create_data_loaders(test_data, args.batch_size)
    
    path = os.path.join(args.model_dir, "model.pth")
    
    for e in range(args.epochs):
        model=train(model, train_loader, loss_criterion, optimizer, hook)

        '''
        TODO: Test the model to see its accuracy
        '''
        test(model, test_loader, hook)
        
        '''
        TODO: Save the trained model
        '''
        torch.save(model, path)
    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--backend', type=str, default='gloo', required=False)
    parser.add_argument('--data', type = str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)
