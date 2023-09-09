# Example of usage: Restricted Boltzmann Machine with continuous-valued outputs
#
# Alvaro Huanay de Dios
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export

import imageio
import numpy as np
import os
import torch
from ebm.optimizers import Adam #From the folder "optimizers" it is importing the Adam optimizer
from ebm.samplers import ParallelTempering #Here it is using parallel tempering
from ebm.models import RBM #From the folder "models" it is importing the RBM 

from torchvision import datasets #Imports datasets from torchvision
#import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

#Álvaro lib:
from matplotlib import pyplot
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision
import torchvision.transforms as T
import tempfile
from ebm.quantum_annealing import quantum_annealing



#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd           = 50          # Number of nodes in the hidden layer
#maximum hidden nodes for quantum annealing (with 49 visible nodes) 60
learning_rate  = 1e-3         # Learning rate
epochs         = 5           # Training epochs (4 minutes per epoch approx)
k              = 1            # Steps of Contrastive Divergence (CD)
k_reconstruct  = 4000         # Steps of iteration during generation
batch_size     = 32           # Batch size
"""
An H5 file is a data file saved in the Hierarchical Data Format (HDF).
It contains multidimensional arrays of scientific data.
"""
gpu            = True         # Use of GPU

loss_list=[]                      #y value to plot the well fitting
x_epoch=np.arange(1,epochs+1) #x value to plot the well fitting
validation_values=[]
dataset_size=7              #set the input dataset size
vneurons=dataset_size**2    #input visible neurons in RBM (Only 784 when dataset_size=28)
classical=True
qubo=True
from dwave.system import DWaveSampler, EmbeddingComposite
sampler_auto = EmbeddingComposite(DWaveSampler( solver={'qpu': True, 'topology__type': 'pegasus'}, token="DEV-6f993718e6825f1b275de666080d60c1753a0bbc"))
#sampler_auto = None
if classical:
    word="Classsical"
    model_dir= str(word)+"_RBM_"+str(epochs)+"epochs_"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.h5"       # Directory for saving last parameters learned
else:
    word="Quantum"
    if qubo==True:
        model_dir= str(word)+"_RBM_QUBO"+str(epochs)+"epochs_"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.h5"       # Directory for saving last parameters learned
    else:
        model_dir= str(word)+"_RBM_ISING"+str(epochs)+"epochs_"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.h5"       # Directory for saving last parameters learned


i=1
w_updates2=[]
w_updates=[]
plot_w=[]
plot_vbias=[]
plot_hbias=[]

plot_vbias_update=[]
plot_hbias_update=[]
plot_w_update=[]

plot_deltah=[]
plot_deltav=[]
plot_deltaw=[]
#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu') #Á: torch device


data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.float) 



#Á: data.size() output: torch.Size([6000,784])

test = datasets.MNIST('mnist',
                      train=False).test_data.type(torch.float)


    
if (dataset_size!=28):
    cut_tensor_data=torch.zeros(data.size(0), 20,20) #Á: Where is going to be stored the data
    cut_tensor_test=torch.zeros(test.size(0), 20,20)
    for i in range(data.size(0)):
        cut_tensor_data[i]=data[i][5:25,5:25] #Á: Now we have a dataset 20x20 (deleted black pixels with no info)  
    for i in range(test.size(0)):
        cut_tensor_test[i]=test[i][5:25,5:25]
    data = T.Resize(size=dataset_size)(data) #Á: Now it is ([60000, dataset_size, dataset_size])
    test = T.Resize(size=dataset_size)(test) #Á: Now it is ([10000, dataset_size, dataset_size])


# -----------------------------------------------------------------------------
# Show an example of the image that will be introduced into the RBM
# -----------------------------------------------------------------------------

print(data[0].size())
example=data[0]
pyplot.imshow(example, cmap=pyplot.get_cmap('gray'))
pyplot.savefig("Input_sample"+str(word)+"_RBM_"+str(epochs)+"epochs"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.png")

#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

data = (data.view((-1, vneurons)) / 255).to(device) 

test = (test.view((-1, vneurons)) / 255).to(device)


print(data.size())
print(test.size())
vis = len(data[0]) 


vbias = torch.log(data.mean(0)/(1 - data.mean(0))).clamp(-20, 20)

#Á: vbias is a tensor of len 784 which is the visible bias b_i
#Á: vbias.size() output: torch.Size([784])

# -----------------------------------------------------------------------------
# Construct RBM
# -----------------------------------------------------------------------------

sampler = ParallelTempering(k=k, #Á:k=1 steps of CD (constrastive divergence)
                            n_chains=100, #Á: Number of parallel chains to be run
                            betas=[0, 0.25, 0.5, 0.75, 1], #Á: List of inverse temperatures of the copies of negative particles
                            #beta: inverse T to cool down the system
                            continuous_output=True) #Á: Optional parameter to output visible activations instead of samples (for continuous-valued functions)

#Á: Here it is using as sampler parallel tempering instead of PCD.

optimizer = Adam(learning_rate)
#Á: The optimizer used is the Adam optimizer to the RBM to learn

#Á: Define the RBM
rbm = RBM(n_visible=vis,
          n_hidden=hidd,
          sampler=sampler,
          optimizer=optimizer,
          device=device,
          vbias=vbias)

pre_trained = os.path.isfile(model_dir)


if pre_trained:
    rbm.load_state_dict(torch.load(model_dir)) 
   
#Á: If it is pretrained, load the values of the weights and biases

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

if not pre_trained:
    #Á: If not pre_trained --> Start the training
    validation = data[:10000] #10.000 images

    for _ in range(epochs):
        
        #Á: Training epochs (In the paper is 100 but here 20. 20 epochs=20 min training)
        
        train_loader = torch.utils.data.DataLoader(data,
                                                   batch_size=batch_size, #Á: tamaño del lote
                                                   shuffle=True) #Á: barajar
        
        rbm.train(train_loader, rbm.state_dict(), classical,_, epochs, qubo, sampler_auto, plot_w, plot_vbias, plot_hbias, plot_deltah, plot_deltav, plot_deltaw, plot_vbias_update, plot_hbias_update, plot_w_update)
        """
        w_updates=rbm.train(train_loader)
        print(len(w_updates[0]))
        
        for i in range(len(w_updates)):
            w_updates2.append(w_updates[i])
            
            
        plt.plot(np.arange(len(w_updates2)), w_updates2, 'ro', markersize=0.2)
        plt.title("weight updates 1 hidden node (mean)")
        plt.xlabel("batch number")
        plt.ylabel("w_update")
        plt.locator_params(axis="y", nbins=20)
        plt.legend(['W_update[0].mean(0).item()'])
        plt.savefig("w_update")
        plt.show()
        """

        # A good measure of well-fitting is the free energy difference
        # between some known and unknown instances.
        #print(rbm.free_energy(validation).size())
        #print(rbm.free_energy(test).size())
        gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)

        
        """
        gap_nomean=rbm.free_energy(validation) - rbm.free_energy(test)
        output=rbm.free_energy(validation)
        labels=rbm.free_energy(test)
        """
        """
        rbm.free_energy(validation):
            
            Computes the free energy for a given state of the visible layer.
            validation: unknown instance 
        
        rbm.free_energy(test)
            
            test: known instance
                
        test.size()==validation.size() True
        type(test)==type(validation) True
        
        """
        
        print('Gap = {}'.format(gap.item()))
        loss_list.append(gap.item())
        print("New gap element added to the list: ", loss_list)
        
        """
        if(_%5==0):
            print("Plotting epoch: "+str(_)+"...")
            x_axis_test_plot=np.arange(1,len(rbm.free_energy(test))+1)
            plt.scatter(x_axis_test_plot,rbm.free_energy(test),c="black", marker="o", s=0.02)
            plt.title("Free energy for a given state\n of the visible layer")
            plt.xlabel("Image vector. 10.000 total images (test set)")
            plt.ylabel("Free energy (test set)")
            plt.locator_params(axis="y", nbins=20)
            plt.legend(["rbm.free_energy(test)"])
            plt.ylim([-350,-50]) #Set limit for the gif and see how the free energy minimizes
            plt.savefig("Epoch_"+str(_)+"_free_energy_no_mean_applied.png")
            plt.show()
        else:
            continue
        """
            
        #Á: gap.item() element of the tensor, which is a escalar in this situation

    torch.save(rbm.state_dict(), model_dir)
    
    #Á: Save the best parameters learned of the weights and biases in model_dir after the training





# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


print('Reconstructing images')
zero = torch.zeros(25, vneurons).to(device)
#Á: Creamos el tensor
images = [zero.cpu().numpy().reshape((5 * dataset_size, 5 * dataset_size))]
#Á: 28x28 1 imagen
#Á: Nuestro gif tiene 25 imágenes (Celdas), por lo tanto, size (5*28,5*28)
for i in range(k_reconstruct):
    #Á: Recuerda: k_reconstruct  = 2000 # Steps of iteration during generation
    zero = sampler.get_h_from_v(zero, rbm.weights, rbm.hbias)
    #Á: Hace una distribución de activación de cuáles de las hidden neurons están activadas (1) o no (0)
    zero = sampler.get_v_from_h(zero, rbm.weights, rbm.vbias)
    #Á Doubt: Backward sampling of the visible layer once the rbm has the weights and vbias trained?
    #Á: Hace una distribución de activación de cuáles de las vissible neurons están activadas (1) o no (0)
    
    #Á: Ahora el sistema tiene vissible y hidden neurons 1 y 0 por nodo. Esta activación representará 1 número.
    #Á: Así estará sampleando las v y h neurons (0,1) durante k_reconstruct veces
    
    if i % 3 == 0:
        datas = zero.data.cpu().numpy().reshape((25, dataset_size, dataset_size))
        #Á: tensor de 25 imágenes, cada una de 28x28
        
        image = np.zeros((5 * dataset_size, 5 * dataset_size))
        #Á: Creamos la imagen para rellenar
        for k in range(5):
            for l in range(5):
                image[dataset_size*k:dataset_size*(k+1), dataset_size*l:dataset_size*(l+1)] = datas[k + 5*l, :, :]
                #Á Doubt: Está metiendo los píxeles en cada imagen datas[25,28,28]
        images.append(image)
                #Á Doubt: Guarda el fotograma en images
imageio.mimsave(str(word)+"_RBM_"+str(epochs)+"epochs"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.gif", images, duration=0.1)
#Á: Crea un gif con la evolución de los fotogramas a medida que k_reconstruct avanza
#Á: Yo ahora me quedo con el último fotograma para analizarlo
pyplot.imshow(images[-1], cmap=pyplot.get_cmap('gray'))
pyplot.savefig(str(word)+"_RBM_"+str(epochs)+"epochs"+str(dataset_size)+"x"+str(dataset_size)+"_"+str(hidd)+"hidden_nodes.png")

#A: Grab each image of this gif:
image=np.zeros((dataset_size,dataset_size))
image[:,:]=datas[5,:,:]
#Á: El índice i de datas[i,:,:] lo que hace es coger cada sub-imagen, el índice i
#Á: representa la imagen por columnas. i=0 es la primera imagen de arriba de todo
#Á: a la izquierda. i=4 es la imagen de abajo de todo a la izquierda.
#Á: Filas y columnas empiezan a contar con 0. i=5 es columna 1, fila 0 de la
#Á: matriz imagen.



#Á: Uncomment when interested in saving generated images
############################
#Á: Grab and save the images
############################
#generated_img=torch.zeros(25,dataset_size,dataset_size).to(device)
#for i in range(image.size(0)):
#    generated_img[i,:,:]=torch.Tensor(datas[i,:,:])
#for i in range(image.size(0)):
    #save_image(image[i], f"C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/resnet_example/image"+str(i)+"_"+str(epochs)+"epochs_k_reconstruct"+str(k_reconstruct)+"_k_"+str(k)+".png")
#path: C:\Users\Alvar\Downloads\TFM_ebm-torch-master\ebm-torch-master_RBM\Alvaro_Project\SubirGitHub\resnet_example


# -----------------------------------------------------------------------------
# Grab the image and convert it into a tensor
# -----------------------------------------------------------------------------
"""
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as pyplot
from PIL import Image
import torchvision.transforms as T


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


pyplot.imshow(imagen, cmap=pyplot.get_cmap('gray'))
"""
############################
#Á: Build the matrix for quantum annealing
############################




# -----------------------------------------------------------------------------
# ResNet
# -----------------------------------------------------------------------------

from resnet_example.resnet18_generic_dataset_size_example import model, features, resnet18, ResNet
#Sólo puedo importar clases
#Si lo importo lo ejecuta

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

"""
Comentario:
    
    Es necesario usar una resnet desde scratch y entrenarla
    en vez de una pre-entrenada porque será necesaria para
    reconocer imágenes 7x7 de debido al quantum annealing.
"""


# ------------------------------------------- #
# Evaluate RBM generated images with resnet18 #
# ------------------------------------------- #

    
    
# ---------------------------------------------------------- #
# Grab tensor images from RBM and introduce them into resnet #
# ---------------------------------------------------------- #

tensor_list=[]
for i in range(len(datas)):
    tensor_list.append(torch.Tensor(datas[i,:,:]))
imagen_tensor=torch.zeros(25, dataset_size, dataset_size).to(device)
for i in range(len(datas)):
    imagen_tensor[i,:,:]=tensor_list[i]
imagen_tensor_input=(imagen_tensor.view((-1, 1, dataset_size,dataset_size)) / 255).to(device)
    


# ------------------------------------------------------ #
#               Normalization
#Note that the RBM does not normalize data once generated:
    
tensor_example=torch.zeros(len(datas),1,dataset_size, dataset_size)
for k in range(len(datas)):
    for i in range(dataset_size):
        for j in range(dataset_size):
            tensor_example[k][0][i][j]=round(imagen_tensor_input[k][0][i][j].item(),4)


X=tensor_example
X -= X.min()
X /= X.max()

results=[]
for i in range(len(datas)):
    logits,probas=model(X.to(device)[i,None])
    for j in range(10):
        print("prob del número ", str(j), " es: ", probas[0][j]*100)
        if (probas[0][j]*100>95):
            results.append(j)
            print("Number found! Number is: "+str(j)+"\n")
        else:
            continue
    print("Next image to analyze: \n")
    
# -------------------------------------- #

#imagen_tensor_input = (imagen_tensor.view((-1, 1,28,28)) / 255).to(device)
accuracy=len(results)/len(datas)*100
if classical:
    print("Accuracy of the classical RBM "+str(dataset_size)+"x"+str(dataset_size)+" with "+str(hidd)+" hidden neurons and "+str(epochs)+" epochs is: {}%".format(accuracy))
else:
    print("Accuracy of the quantum RBM"+str(dataset_size)+"x"+str(dataset_size)+" with "+str(hidd)+" hidden neurons and "+str(epochs)+" epochs is: {}%".format(accuracy))
    
"""
model.eval()
for i in range(len(datas)):
    logits,probas=model(imagen_tensor_input.to(device)[i,None])
    for j in range(10):
        print("prob del número ", str(j), " es: ", probas[0][j]*100)
        if (probas[0][j]*100>95):
            results.append(j)
            print("Number found! Number is: "+str(j)+"\n")
        else:
            continue
    print("Next image to analyze: \n")

#Á: Test resnet

misclassified=[]
accuracy=len(results)/25
print("accuracy of this set is: ", accuracy)
#Á: accuracy is the ones that have prob>0.95 divided by the total dataset
"""