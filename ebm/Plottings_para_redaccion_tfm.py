# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:20:05 2022

@author: Alvar
"""

"""

print("Plotting the gap\n")

plt.plot(x_epoch, loss, 'r--o')
plt.title("Mean of the difference between\n validation and test free energies")
plt.xlabel("Epoch")
plt.ylabel("Gap")
plt.locator_params(axis="y", nbins=20)
plt.legend(['gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)'])
plt.savefig("gapVSepoch_mean_applied.png")
plt.show()



print("Plotting the LAST validation batch (last iteration training result)\n")

x_axis_validation_plot=np.arange(1,len(rbm.free_energy(validation))+1)
plt.plot(x_axis_validation_plot,rbm.free_energy(validation),"go", markersize=0.2)
plt.title("Validation values per batch\n No applied mean")
plt.xlabel("Datapoint for free energy")
plt.ylabel("Free energy of the validation")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(validation)"])
plt.savefig("Last_Epoch_Validation_Batch_no_mean_applied.png")
plt.show()

print("Plotting the LAST test batch (last iteration training result)\n")

x_axis_test_plot=np.arange(1,len(rbm.free_energy(test))+1)
plt.scatter(x_axis_test_plot,rbm.free_energy(test),c="black", marker="o", s=0.02)
plt.title("Test values per batch\n No applied mean")
plt.xlabel("x datapoint for the test free energy")
plt.ylabel("Free energy of the test")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(test)"])
plt.savefig("Last_Epoch_test_Batch_no_mean_applied.png")
plt.show()

print("Plotting the LAST validation batch difference\n between validation and test")
x_gap_nomean=np.arange(1,len(gap_nomean)+1)
plt.plot(x_gap_nomean,gap_nomean,"bo", markersize=0.2)
plt.title("Difference\n rbm.free_energy(validation) - rbm.free_energy(test) \n in LAST epoch (10.000 images) how the free energy differs")
plt.xlabel("Image number")
plt.ylabel("Free energy of validation-test (no mean)")
plt.locator_params(axis="y", nbins=20)
plt.legend(["rbm.free_energy(validation) - rbm.free_energy(test)"])
plt.savefig("Last_Epoch_Validation-Test_Batch_no_mean_applied.png")
plt.show()
"""


"""
#All weights of the rbm after training (completely updated after training)
#This can be similarly applied to hbias and vbias after training
rbmweights=[]
for i in range(rbm.weights.size(0)):
    for j in range(rbm.weights.size(1)):
        rbmweights.append(rbm.weights[i][j].item())
plt.plot(np.arange(len(rbmweights)), rbmweights, 'ro', markersize=0.2)
plt.title("All weights of the rbm after training\n (completely updated after training)")
plt.xlabel("weight_i")
plt.ylabel("value")
plt.locator_params(axis="y", nbins=20)
plt.legend(['rbm.weights[i][j].item()'])
plt.savefig("weight")
plt.show()
"""


"""
print("Visualize the real image")
def show_adn_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f,npimg)
show_adn_save("real_image",make_grid(validation.view(batch_size,1,28,28).data)) #antes era -1=32
"""

# ------------------------------------------------------------------------#


"""

from dwave.system import DWaveSampler, EmbeddingComposite
sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'})

sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
"""

"""
#Á: This doesn´t work. Sampler needs a dictionary, not a matrix

matrix=torch.zeros(1084,1084).to(device) #300+784=1084, the matrix must be squared 1084x1084
for i in range(len(rbm.state_dict()["vbias"])):
    matrix[i,i]=rbm.state_dict()["vbias"][i].item()
    #Á: Grab each vbias element and introduce in the first diagonal
for i in range(len(rbm.state_dict()["hbias"])):
    matrix[len(rbm.state_dict()["vbias"])+i,len(rbm.state_dict()["vbias"])+i]=rbm.state_dict()["hbias"][i].item()
    #Á: Grab each hbias element and introduce in the continuation of the diagonal
r=0
k=0
t=1
for i in range(300):
    for j in range(784):
        matrix[r][k+t]=rbm.state_dict()["weights"][i][j].item()
        if (i==299 and j==783):
        #Á: Mira, los weights no llenan toda la upper matrix
            print("row of matriz is: ", r)
            print("column of matriz is: ", k+t)
            print("last element of weights is: ", rbm.state_dict()["weights"][i][j].item())
            print("last element of weights in matriz is: ", matrix[r][k+t])
        k=k+1 #Á: Next column in the matrix
        if (k+t==1083): #Á: If we arrive to the last column in matrix
            r=r+1 #Á: Pasa siguiente fila
            k=0   #Á: vuelve a empezar columna
            t=t+1 #Á: empieza en una columna más que la anterior
        else:
            continue
"""

"""

qubit_biases={} #Á: Create dictionary
values=[]
for i in range(len(rbm.state_dict()["vbias"])):
    values.append(rbm.state_dict()["vbias"][i].item()) #Á: Store the value in the list
for i in range(len(rbm.state_dict()["hbias"])):
    values.append(rbm.state_dict()["hbias"][i].item()) #Á: Store the value in the list
    
for i in range(1084):
    qubit_biases[(i,i)]=values[i]   #Create the keys as tuples and assign the value as each value

qubit_weights={} #Á: Create a dictionary
weights=[]
for i in range(300):
    for j in range(784):
        weights.append(rbm.state_dict()["weights"][i][j].item()) #Á: Store all the values in a list
t=1
k=0
for j in range(1084):
    for i in range(1084):
        if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
            break
            break
        qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
        k=k+1
    t=t+1

coupler_strengths=qubit_weights
Q = {**qubit_biases, **coupler_strengths}
sampleset = sampler_auto.sample_qubo(Q, num_reads=100)
#sampleset = sampler_manual.sample_qubo(Q, num_reads=5000)
print(sampleset)

"""


# ------------------------ QA ------------------------------ #

    """
    t=1
    k=0
    #matriz 1084x1084
    for j in range(1084): #para cada fila de la matriz
        for i in range(1084): #para cada columna de la matriz
            if (i+t==1084): #si llega a la última columna de la matriz
                break #exit i loop to add +1 to j y t para pasar a la siguiente fila por encima de la diagonal
            qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
            k=k+1
            if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
                print("Se acabaron los elementos de weights, quedarán huecos.\n Último elemento del diccionario es fila "+str(j)+"columna "+str(i+t))
                break
        if (k==len(weights)):
            break
        t=t+1
    """
    
    """
    
    
    t=1
    k=0
    for j in range(rowscolumns):
        for i in range(rowscolumns):
            if (k==len(weights)): #Á: If all elements in the list are added to the dictionary exit both loops
                break
                break
            qubit_weights[(j,i+t)]=weights[k] #Á: Add the element list to the dictionary
            k=k+1
        t=t+1
    """
    
    
    
            """
        h_df=torch.zeros(hbias.size(0),num_reads)
        h_model=torch.zeros(hbias.size(0))
        for j in range(h_df.size(0)):
            for i in range(h_df.size(1)):
                h_df[j][i]=qubo_df.iloc[i,j+vbias.size(0)]
            h_df[j]=1/h_df.size(1)*np.cumsum(h_df[j])
            h_model[j]=h_df[j][-1]
        """
        
        
        """
        
        v_df=torch.zeros(vbias.size(0),num_reads) #Á: Creamos un dataframe con toda la información de vbias necesaria
        v_model=torch.zeros(vbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(v_df.size(0)): #Á: Para cada fila del dataframe (49)
            for i in range(v_df.size(1)): #Á: Para cada columna del dataframe (5000)
                v_df[j][i]=ising_df.iloc[i,j] #Á: Coge cada elemento del Ising sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            v_df[j]=1/v_df.size(1)*np.cumsum(v_df[j]) #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
            v_model[j]=v_df[j][-1] #Á: Este será v_model, la negative phase del visible bias. Donde el gradiente de la loss: update=vpos-vneg. v_model=veng
        
        #h_model:
            
        h_df=torch.zeros(hbias.size(0),num_reads)
        h_model=torch.zeros(hbias.size(0))
        for j in range(h_df.size(0)):
            for i in range(h_df.size(1)):
                h_df[j][i]=ising_df.iloc[i,j+vbias.size(0)]
            h_df[j]=1/h_df.size(1)*np.cumsum(h_df[j])
            h_model[j]=h_df[j][-1]
    
        #w_model:
        
        w_model=torch.zeros(weights_tensor.size(0),weights_tensor.size(1))
        #300*49 avg elements que vamos a calcular para la matriz
        for j in range(weights_tensor.size(0)):
            for i in range(weights_tensor.size(1)):
                elements_sum=ising_df.iloc[:,i]*ising_df.iloc[:,j+weights_tensor.size(1)]
                avg_element=1/num_reads*np.cumsum(elements_sum)
                w_model[j][i]=avg_element.iloc[-1]
        
        """
        
# ---------------------------------------------------------- #
