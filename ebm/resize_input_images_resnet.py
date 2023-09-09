# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:14:36 2022

@author: Alvar

This code grabs the images produced by the RBM (28x28)
and resizes it into tensors ([1,1,7,7]) to evaluate in
the resnet.
"""

import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision.transforms as T

gpu=True
device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device

imagen = Image.open('7_number.png')
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])
#Á: Transform image to tensor
img_tensor = transform(imagen) #Á: Now it is: torch.Size([3, 28, 28])
imagen_resnet=img_tensor[0:1] #Now it is: torch.Size([1, 28, 28])
imagen_resnet = torch.tensor(np.array([imagen_resnet.numpy()]))
#Á: Now it is: torch.Size([1, 1, 28, 28]) to introduce into the resnet
imagen_resnet_input = (imagen_resnet.view((-1, 1,28,28)) / 255).to(device) #Á: Notmalize and introduce into device


resized_img = T.Resize(size=7)(imagen_resnet_input) #Á: Resize ([1,1,28,28]) to ([1,1,7,7]) for QPU
#Original image
pyplot.imshow(imagen, cmap=pyplot.get_cmap('gray'))
#Resized image
pyplot.imshow(resized_img[0][0], cmap=pyplot.get_cmap('gray'))