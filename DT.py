#%%

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
from sklearn import tree
from sklearn import ensemble

from helpers import plot
import torch.nn.init as init
import re
import xgboost as xgb



#%%
with open("/Users/danielegana/Dropbox (PI)/ML/code/DT/inputs.txt", 'r') as file:
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
# For decision trees I train different ones, with differend random seeds, to check that the accuracy
# doesn't depend much on the random seed. I get it doesn't, differently from CNNs.
# I get overall 80% classification accuracy. It's not that bad. It improves sligthly with max_depth pruning
numseeds=2
accuracyvector=np.zeros(numseeds+1)

for x in range(numseeds+1):
    model=tree.DecisionTreeClassifier(criterion='entropy',random_state=seed+x)
    size=training_data.data.size()[1:][0]*training_data.data.size()[1:][1]
    np.random.seed(seed+x)
    model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())
    predictions=model.predict(test_data.data.view(-1,size).numpy())
    counter=0
    for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
    print("Accuracy for seed=",x,"is ",accuracy)    
    accuracyvector[x]=accuracy

# The code below is if you want to train the tree only for one random seed

#%%
model=tree.DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=15,min_samples_leaf=10)
model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())

#%%
counter=0
predictions=model.predict(test_data.data.view(-1,size).numpy())
for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
print("Accuracy is=",accuracy) 
# %%

# Now I try random forest, my personal favorite! I get accuracies of order 88%, which is pretty great considerin
# no data preprocessing was done, and no GPUs were used. Increasing the number of estimators to 500 doesn't really help,
# and playing with max_depth helps only a bit.

#%%
model=ensemble.RandomForestClassifier(n_estimators=200)
model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())

#%%
counter=0
predictions=model.predict(test_data.data.view(-1,size).numpy())
for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
print("Accuracy is=",accuracy) 
# %%


# Now what about gradient boosted classifiers. It just takes forever to train so I don't do it.
#%%
model=ensemble.GradientBoostingClassifier()
model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())

#%%
counter=0
predictions=model.predict(test_data.data.view(-1,size).numpy())
for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
print("Accuracy is=",accuracy) 
# %%

# Let's try now the bagging classifier. It gives 85%, not bad
#%%
model=ensemble.BaggingClassifier()
model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())

#%% 
counter=0
predictions=model.predict(test_data.data.view(-1,size).numpy())
for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
print("Accuracy is=",accuracy) 
# %%


# Let's try now XGBoost, maybe that's faster. It is and wow, out of the box gives 90%.
#%%
model=xgb.XGBClassifier(learning_rate=0.01,n_estimators=200)
model.fit(training_data.data.view(-1,size).numpy(),training_data.targets.numpy())

#%% 
counter=0
predictions=model.predict(test_data.data.view(-1,size).numpy())
for z,y in enumerate(test_data):
        if(y[1]==predictions[z]):
            counter+=1
            accuracy=counter/len(test_data)*100
print("Accuracy is=",accuracy) 
# %%
