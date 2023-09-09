# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:46:27 2022

@author: Álvaro Huanay de Dios

This is the code for quantum annealing. This fragment creates the Q
matrix, uses automated embedding and samples.
"""

#libraries:
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import torch
import time


def quantum_annealing(rbm_state_dict,batch_num , qubo, sampler_auto):
    #qubo=bool para saber si queremos hacer el sampling con qubo model o ising model
    
    #rowscolumns=rbm.state_dict()["vbias"].size(0)+rbm.state_dict()["hbias"].size(0)

    vbias=rbm_state_dict["vbias"]
    hbias=rbm_state_dict["hbias"]
    weights_tensor=rbm_state_dict["weights"]
    
    beta=1.5
    # -------------- #
    # qubit biases   #\
    # -------------- #
    
    qubit_biases={} #Á: Create dictionary
    values=[]
    for i in range(len(vbias)):
        values.append(vbias[i].item()) #Á: Store the value in the list
    for i in range(len(hbias)):
        values.append(hbias[i].item()) #Á: Store the value in the list
    
    for i in range(vbias.size(0)+hbias.size(0)):
        qubit_biases[(i,i)]=(1/beta)*values[i]   #Create the keys as tuples and assign the value as each value
    
    # ----------------- # 
    # coupler_strengths #
    # ----------------- #
    
    qubit_weights={} #Á: Create a dictionary
    weights=[]
    k=0
    for i in range(len(hbias)):
        for j in range(len(vbias)):
            weights.append(weights_tensor[i][j].item()) #Á: Store all the values in a list
            
    for j in range(len(vbias)): #para cada fila de la matriz
        for i in range(len(hbias)): #para cada columna de la matriz
            qubit_weights[(j,i+len(vbias))]=(1/beta)*weights[k] #Á: Add the element list to the dictionary
            k=k+1
    

    
    # ------------------------------------------- #
    # Build the Q matrix, sample and compute avgs #
    # ------------------------------------------- #
    
    coupler_strengths=qubit_weights
    Q = {**qubit_biases, **coupler_strengths}
    num_reads=500
    
    #sampler_manual = DWaveSampler(solver={'topology__type': 'chimera'}, token="DEV-0467747fa9018fbad8c93ccdc0eab18075f8e5bf")    
    
    """
    
    #Artur proposition:
    #Find the index of the minimum energy in the Ising model:
        
    ising_df.iloc[ising_df["energy"].idxmin()]
    #Los valores de la diagonal son los vbias y hbias. Han de ser multiplicaciones +1*-1 y sus permutaciones
    #La configuración vbias hbias que me da el mínimo es la correcta para ese annealing.
    #5000 samplings, 5000 energías, cojo la mínima energía, y la configuración vbias + hbias y hago su avg product.
    """
    
    #avgs
    
    if (qubo==True):
        
    # -------------------- #
    # avgs with QUBO model #
    # -------------------- #
        
    
        #Qubo model
        print("Sampling...\n")
        start_sampling = time.time()
        sampleset_qubo = sampler_auto.sample_qubo(Q, num_reads=num_reads) #Á: Auto sampler mode
        end_sampling = time.time()
        print("Dwave process to send + sample + receive the data is: ", end_sampling-start_sampling)
        #sampleset = sampler_manual.sample_qubo(Q, num_reads=5000) #Á: Manual sampler mode
        #print(sampleset_qubo)
        #print("Creating pandas dataframe...\n")
        qubo_df=sampleset_qubo.to_pandas_dataframe()
        #print("Plotting energy per sample in the annealing QUBO model...\n")
        #plt.title("Energy per sample in the annealing Qubo model")
        #plt.xlabel("Sample number")
        #plt.ylabel("Energy of the sample")
        #qubo_df["energy"].plot(c="black", marker="o", linewidth=0.02, markersize=0.05);
        
        print("Computing negative phases...\n")
        #start_negphase=timer()
        
        #v_model:
            
        #v_df=torch.zeros(vbias.size(0),num_reads) #Á: Creamos un dataframe con toda la información de vbias necesaria
        v_model=torch.zeros(vbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(v_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            v_model[j]=1/num_reads*np.cumsum(qubo_df.iloc[:,j]*qubo_df.iloc[:]["num_occurrences"]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
                #Á: Este será v_model, la negative phase del visible bias. Donde el gradiente de la loss: update=vpos-vneg. v_model=veng
        #print(v_model)
        #h_model:

        
        h_model=torch.zeros(hbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(h_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            h_model[j]=1/num_reads*np.cumsum(qubo_df.iloc[:,j+v_model.size(0)]*qubo_df.iloc[:]["num_occurrences"]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
        #print(h_model)
        #w_model:
            
        
        w_model=torch.zeros(weights_tensor.size(0),weights_tensor.size(1))
        #300*49 avg elements que vamos a calcular para la matriz
        for j in range(weights_tensor.size(0)):
            for i in range(weights_tensor.size(1)):
                w_model[j][i]=1/num_reads*np.cumsum(qubo_df.iloc[:,i]*qubo_df.iloc[:,j+weights_tensor.size(1)]*qubo_df.iloc[:]["num_occurrences"]).iloc[-1]
    
        #print(w_model)
        #end_negphase=timer()
        #print("Negative phases computed. Runtime: \n", end_negphase-start_negphase)
        #print(v_model)
        #print(h_model)
        #print(w_model)
        
    # ---------------------- #
    # avgs with Ising  model #
    # ---------------------- #
    
    else:
        
        #Ising model
    
        sampleset_ising= sampler_auto.sample_ising(qubit_biases, coupler_strengths, num_reads=num_reads) #Á: Auto sampler mode
        #print(sampleset_ising)
        ising_df=sampleset_ising.to_pandas_dataframe()
        """
        plt.title("Energy per sample in the annealing Ising model")
        plt.xlabel("Sample number")
        plt.ylabel("Energy of the sample")
        ising_df["energy"].plot(c="blue", marker="o", linewidth=0.02, markersize=0.05);
        """
        #v_model:       
        
        v_model=torch.zeros(vbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(v_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            v_model[j]=1/num_reads*np.cumsum(ising_df.iloc[:,j]*ising_df.iloc[:]["num_occurrences"]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
                #Á: Este será v_model, la negative phase del visible bias. Donde el gradiente de la loss: update=vpos-vneg. v_model=veng
        #print(v_model)
        #h_model:

        
        h_model=torch.zeros(hbias.size(0))   #Á: Creamos el tensor "visible bias negative phase"
        for j in range(h_model.size(0)): #Á: Para cada fila del dataframe (49)
            #for i in range(num_reads): #Á: Para cada columna del dataframe (5000) #Á: Coge cada elemento del qubo sampling que corresponde a vbias variable (j:0-48, i:0-5000) y mételo en el dataframe
            h_model[j]=1/num_reads*np.cumsum(ising_df.iloc[:,j+v_model.size(0)]*ising_df.iloc[:]["num_occurrences"]).iloc[-1] #Á: Ahora por cada fila de v_df nos interesa únicamente el elemento [j][-1] (última columna de la row j)
        #print(h_model)
        #w_model:
            
        
        w_model=torch.zeros(weights_tensor.size(0),weights_tensor.size(1))
        #300*49 avg elements que vamos a calcular para la matriz
        for j in range(weights_tensor.size(0)):
            for i in range(weights_tensor.size(1)):
                w_model[j][i]=1/num_reads*np.cumsum(ising_df.iloc[:,i]*ising_df.iloc[:,j+weights_tensor.size(1)]*ising_df.iloc[:]["num_occurrences"]).iloc[-1]
                
        print(v_model.size())
        print(h_model.size())
        print(w_model.size())

    return v_model, h_model, w_model
    #return v_model