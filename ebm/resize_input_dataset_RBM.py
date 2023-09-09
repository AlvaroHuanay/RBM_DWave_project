# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:49:43 2022

@author: Alvar

This code grabs the MNIST dataset (28,28) and transforms the input
tensors into tensor dataset (7,7)
"""

import numpy as np
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision
import torchvision.transforms as T
import tempfile


gpu=True
device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device

#Á: Grab data
data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.float)
#Á: Resize it
resized_data = T.Resize(size=7)(data) #Á: Now it is ([60000,7,7])

#Original dataset
pyplot.imshow(data[0], cmap=pyplot.get_cmap('gray'))
#Resized dataset
pyplot.imshow(resized_data[0], cmap=pyplot.get_cmap('gray'))