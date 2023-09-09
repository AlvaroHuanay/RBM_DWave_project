# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:24:52 2022

@author: Álvaro Huanay de Dios
"""

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as T
from matplotlib import pyplot


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 1

# Architecture
dataset_size=7
NUM_FEATURES = dataset_size*dataset_size
NUM_CLASSES = 10

# Other
gpu=True
DEVICE = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device
device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device
GRAYSCALE = True
model_dir      = "resnet"+str(dataset_size)+"x"+str(dataset_size)+".h5"  
cost_list=[]
batch_list=[]

##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)



# -----------------------------------------------------------------------------
# Show an example of the image that will be introduced into the RBM
# -----------------------------------------------------------------------------

pre_trained = os.path.isfile(model_dir)


# Checking the dataset
for images, labels in train_loader: 
    if (dataset_size!=28):
        cut_images_data=torch.zeros(images.size(0),1, 20,20) #Á: Where is going to be stored the data
        for i in range(images.size(0)):
            cut_images_data[i][0]=images[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
        images=T.Resize(size=dataset_size)(cut_images_data) #Á: Now it is ([60000, dataset_size, dataset_size])
    else:
        continue
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


    
    
#print(images[0].size())
#example=images[0][0]
#pyplot.imshow(example, cmap=pyplot.get_cmap('gray'))
#imshow only shows the last imshow (therefore only will be shown the image at the end of the code)
#pyplot.savefig("Input_resnet_sample"+str(dataset_size)+"x"+str(dataset_size)+".png")



device = torch.device(DEVICE)
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        if (dataset_size!=28):
            cut_images_data=torch.zeros(x.size(0), 1, 20,20) #Á: Where is going to be stored the data
            for i in range(x.size(0)):
                cut_images_data[i][0]=x[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
            x=T.Resize(size=dataset_size)(cut_images_data).to(device) #Á: Now it is ([60000, dataset_size, dataset_size])
        else:
            x=x.to(device)
        #x = x.to(device)
        y = y.to(device)
        break
    
    
# ----------------------- #
# Definition of the model #
# ----------------------- #


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model

torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        if (dataset_size!=28):
            cut_images_data=torch.zeros(features.size(0),1, 20,20) #Á: Where is going to be stored the data
            for i in range(features.size(0)):
                cut_images_data[i][0]=features[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
            features=T.Resize(size=dataset_size)(cut_images_data).to(device) #Á: Now it is ([60000, dataset_size, dataset_size])
        #features=T.Resize(size=dataset_size)(features).to(device)
        else:
            features=features.to(device)
        #features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


# ----------------------- #
# Check if trained or not #
# ----------------------- #



if pre_trained:
    model.load_state_dict(torch.load(model_dir))
    
        

# ----------------------- #
# Trainning               #
# ----------------------- #


if not pre_trained:

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            
            if (dataset_size!=28):
                cut_images_data=torch.zeros(features.size(0),1, 20,20) #Á: Where is going to be stored the data
                for i in range(features.size(0)):
                    cut_images_data[i][0]=features[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
                features=T.Resize(size=dataset_size)(cut_images_data).to(device) #Á: Now it is ([60000, dataset_size, dataset_size])
            #features=T.Resize(size=dataset_size)(features).to(device)
            else:
                features=features.to(device)

            #features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            #print(features.size())
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 20:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                       %(epoch+1, NUM_EPOCHS, batch_idx, 
                         len(train_loader), cost))
                cost_list.append(cost.item())
                batch_list.append(batch_idx)
    
            
    
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                  epoch+1, NUM_EPOCHS, 
                  compute_accuracy(model, train_loader, device=DEVICE)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    torch.save(model.state_dict(), model_dir)
    

    
# ----------------------- #
# Evaluation of the model #
# ----------------------- #

    
with torch.set_grad_enabled(False): # save memory during inference
   print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    
    
for batch_idx, (features, targets) in enumerate(test_loader):

    if (dataset_size!=28):
        cut_images_data=torch.zeros(features.size(0),1, 20,20) #Á: Where is going to be stored the data
        for i in range(features.size(0)):
            cut_images_data[i][0]=features[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
        features=T.Resize(size=dataset_size)(cut_images_data).to(device) #Á: Now it is ([60000, dataset_size, dataset_size])
    #features=T.Resize(size=dataset_size)(features).to(device)
    else:
        features=features.to(device)

    #features_test_original=features
    #features=T.Resize(size=dataset_size)(features)
    #features = features
    targets = targets
    break


"""
if (dataset_size!=28):
    print("At the end features is", features.size())
    cut_images_data=torch.zeros(features.size(0),1, 20,20) #Á: Where is going to be stored the data
    for i in range(features.size(0)):
        cut_images_data[i][0]=features[i][0][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
    features=T.Resize(size=dataset_size)(cut_images_data).to(device) #Á: Now it is ([60000, dataset_size, dataset_size])
#features=T.Resize(size=dataset_size)(features).to(device)
else:
    features=features.to(device)
"""
model.eval()
#Uncomment this. Example to determine number 7:
"""
print(features[0].size())
example=features[0][0]
pyplot.imshow(example, cmap=pyplot.get_cmap('gray'))
#imshow only shows the last imshow (therefore only will be shown the image at the end of the code)
pyplot.savefig("Input_resnet_test_sample"+str(dataset_size)+"x"+str(dataset_size)+".png")

    
nhwc_img = np.transpose(features[0], axes=(1, 2, 0))
nhw_img = np.squeeze(nhwc_img.numpy(), axis=2)
plt.imshow(nhw_img, cmap='Greys');
print("size input resnet: ", features.to(device)[0,None].size())

model.eval()
for i in range(10):
    logits, probas = model(features.to(device)[0, None])
    print("Probability of being a "+str(i)+ " %.2f%%" % (probas[0][i]*100))
""" 
    
"""
plt.plot(batch_list, cost_list, 'r--o', markersize=0.2)
plt.title("Resnet18 cost per batch")
plt.xlabel("Batch")
plt.ylabel("Cost")
plt.legend(["Resnet18 28x28", "Resnet 18 7x7"])
plt.locator_params(axis="y", nbins=10)
#plt.legend(['W_update[0].mean(0).item()'])
plt.savefig("Resnet18_"+str(dataset_size)+"x"+str(dataset_size)+"cost")
"""