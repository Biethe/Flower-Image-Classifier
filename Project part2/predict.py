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

parser = argparse.ArgumentParser()
parser.add_argument("image_path",
                    help="the path to the image")
parser.add_argument("checkpoint",
                    help="checkpoint of a trained model")
parser.add_argument("--top_k", type=int,
                    help='number of top classes')
parser.add_argument("--category_names", 
                    help="path to file in which is stored the equivalent names to the labels")
parser.add_argument("--gpu", action="store_true",
                    help="number of units in your hidden layer")
args = parser.parse_args()

# Dataset for label matching
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    device = torch.device("cuda" if args.gpu else "cpu")
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    model_state_dict = checkpoint['state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    class_to_idx = checkpoint['class_to_idx']

    return model_state_dict, optimizer_state_dict, epoch, loss, class_to_idx, input_size, output_size, hidden_layers

model_state_dict, optimizer_state_dict, epoch, loss, class_to_idx, input_size, output_size, hidden_layers = load_checkpoint('checkpoint.pth')
model.load_state_dict(model_state_dict)

def process_image(image_path):
    # Define the transformation for the input image
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open and preprocess the image
    image = Image.open(image_path)
    image = image_transform(image)
    np_image = np.array(image)

    return np_image


def predict(image_path, model_checkpoint):
    # Process the input image
    image = torch.tensor(process_image(image_path))
    model_state_dict, optimizer_state_dict, epoch, loss, class_to_idx, input_size, output_size, hidden_inputs = load_checkpoint('checkpoint.pth')
    
    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_inputs),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_inputs, 102),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier in the pre-trained model
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model_state_dict, optimizer_state_dict, epoch, loss, class_to_idx, input_size, output_size, hidden_layers = load_checkpoint('checkpoint.pth')
    model.load_state_dict(model_state_dict)
    model.class_to_idx = class_to_idx

    # Move the model to CPU (if it's on GPU)
    device = device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model.forward(torch.unsqueeze(image, 0))
    if args.top_k:
        topk = args.top_k
    else:
        topk = 5 
    # Calculate probabilities and classes
    probabilities = torch.exp(output)
    top_probabilities, top_classes = probabilities.topk(topk, dim=1)

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = np.array(top_classes.cpu().reshape(topk))
    # top_classes = [idx_to_class[idx] for idx in top_classes]
    return top_probabilities, top_classes

image_path = args.image_path
checkpoint_path = args.checkpoint

# Test out your network!
image = process_image(image_path)
top_probs, top_classes = predict(image_path, checkpoint_path)
classes = [cat_to_name[idx] for idx in top_classes]
print(f"The following list refers to the top {topk} classes that the model predicts for your image and their respective probability")
for i in range(len(classes):
    print(f"{classes[i]}, {top_probs[i]}")

