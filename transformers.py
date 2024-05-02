#%%

#Now I'll try to do the same with a transformer
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from numpy import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from helpers import plot
import torch.nn.init as init
import re
import torchvision.models as models

from vit_pytorch import ViT
from vit_pytorch import SimpleViT
from vit_pytorch.na_vit import NaViT




#%%
with open("inputs.txt", 'r') as file:
        for line in file:
            matchbatch = re.search(r'batchsize\s*=\s*(\d+)', line)
            matchepoch = re.search(r'epochs\s*=\s*(\d+)', line)
            matchseed = re.search(r'seed\s*=\s*(\d+)', line)
            if matchbatch:
                batch_size=int(matchbatch.group(1))
            if matchepoch:
                epochs=int(matchepoch.group(1))
            if matchseed:
                seed=int(matchseed.group(1))
file.close()

#%%

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
#%%

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#%%
# Define model as a class. We're creating a new class mynet.
# mynet inherits features from a base class nn.Module that allows it to perform GPU acceleration and others

#%%
#Defines the function to train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #Initializes the training mode of the model
    model.train()
    #enumerate creates a tuple index,data. so batch gets the index number
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # The loss is computed against the known class label. y is an integer, pred is a 10-dimensional vector
        # with the 10 classes. 
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        #optimizer.zero_grad() zeroes out the gradient after one pass. this is to 
        #avoid accumulating gradients, which is the standard behavior
        optimizer.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}]")

# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


numseeds=0
accuracyvector=np.zeros(numseeds+1)
flag=0
# %% 
for x in range(numseeds+1):
    print("Seed run =",x)
    torch.manual_seed(seed+x)
    np.random.seed(seed+x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = SimpleViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 10,
    dim = 100,
    depth = 8,
    heads = 32,
    mlp_dim = 100,
    channels=1).to(device)
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy=100*test(test_dataloader, model, loss_fn)
        #A little code to control the learning rate
        print("Done!")
    accuracyvector[x]=accuracy


# %%
