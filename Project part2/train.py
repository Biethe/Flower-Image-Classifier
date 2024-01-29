import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.autograd import Variable
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("data_directory",
                    help="the directory wherein your data is stored ")
parser.add_argument("--save_dir",
                    help="the directory wherein you want to save your checkpoints")
parser.add_argument("--arch", choices = ['VGG', 'Densenet'],
                    help="neural network architecture")
parser.add_argument("--learning_rate", type = float,
                    help="learning_rate of the network")
parser.add_argument("--hidden_units", type = int,
                    help="number of units in your hidden layer")
parser.add_argument("--epochs", type = int,
                    help="number of epochs chosen for the neural network training")
parser.add_argument("--gpu", action="store_true",
                    help="number of units in your hidden layer")
args = parser.parse_args()                   

# Check torch version and CUDA status if GPU is enabled.
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled.
                    
                    
if args.data_directory:
                    data_dir = args.data_directory
else:
                    data_dir = 'flower'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# transforms for the training, validation, and testing sets
data_transforms = {
   'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

   'test': transforms.Compose([
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

}

# TODO: Load the datasets with ImageFolder
image_datasets = {
   'train': datasets.ImageFolder(train_dir,
       transform=data_transforms['train']
   ),
    'valid': datasets.ImageFolder(valid_dir,
       transform=data_transforms['valid']
   ),

   'test': datasets.ImageFolder(test_dir,
       transform=data_transforms['test']
   )

}

# : Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
   x: torch.utils.data.DataLoader(
       image_datasets[x], batch_size= 32, shuffle=True)
   for x in ['train', 'valid', 'test']
}


if args.hidden_units:
                    hidden_units = args.hidden_units
else:
                    hidden_units = 1024
# Import of the pre_trained convolution model
if args.arch=='VGG':
                    model = models.vgg16(pretrained=True)
                    # Freeze parameters so we don't backprop through 
                    for param in model.parameters():
                        param.requires_grad = False

                    # Define a new classifier
                    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(hidden_units, 102),
                                                nn.LogSoftmax(dim=1))

                                                # Replace the classifier in the pre-trained model
                                                model.classifier = classifier
if args.arch=='Densenet':
                    model = models.densenet121(pretrained=True)
                    # Freeze parameters so we don't backprop through 
                    for param in model.parameters():
                        param.requires_grad = False

                    # Define a new classifier
                    classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(hidden_units, 102),
                                                nn.LogSoftmax(dim=1))

                    # Replace the classifier in the pre-trained model
                    model.classifier = classifier
                    inputs = 1024

else:
                    model = models.vgg19_bn(pretrained=True)
                    # Freeze parameters so we don't backprop through 
                    for param in model.parameters():
                        param.requires_grad = False

                    # Define a new classifier
                    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(hidden_units, 102),
                                                nn.LogSoftmax(dim=1))

                    # Replace the classifier in the pre-trained model
                    model.classifier = classifier
                    inputs = 25088

    
criterion = nn.NLLLoss()
if args.learning_rate:
                    lr = args.learning_rate
else:
                    lr = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
# Define loss function and optimizer

# Move model to GPU if available
device = torch.device("cuda" if args.gpu else "cpu")
model.to(device)

# Train the classifier
if args.epochs:
                    epochs = args.epochs
else:
                    epochs = 5
print_every = 50
running_loss = 0
for epoch in range(epochs):
    for step, (inputs, labels) in enumerate(dataloaders['train'], 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if step % print_every == 0:
            # Print training loss every `print_every` steps
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Step {step}/{len(dataloaders['train'])}.. "
                  f"Training loss: {running_loss/print_every:.3f}")
            running_loss = 0

# Validation loss
model.eval()
validation_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in dataloaders['valid']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        batch_loss = criterion(outputs, labels)
        validation_loss += batch_loss.item()
        # Calculate accuracy
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Epoch {epoch+1}/{epochs}.. "
      f"Training loss: {running_loss/print_every:.3f}.. "
      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
running_loss = 0
model.train()

# TODO: Do validation on the test set
def validation(model, testloader, criterion):
    # Use GPU in case it's available
    device = torch.device("cuda:0" if args.gpu else "cpu")
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")
        return test_loss/len(testloader), accuracy/len(testloader)

validation(model, dataloaders['test'], criterion)

# TODO: Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx


checkpoint = {'input_size': inputs,
              'output_size': 102,
              "hidden_layers": hidden_units,
              'state_dict': model.state_dict(),
              'epoch': epoch,
              'loss': running_loss / print_every,
              'optimizer_state_dict': optimizer.state_dict,
              'class_to_idx': image_datasets['train'].class_to_idx}
if args.save_dir:
                    torch.save(checkpoint, args.save_dir)
else:
                   torch.save(checkpoint, 'checkpoint.pth') 
