# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:46:29 2022

@author: Alvaro Huanay de Dios
"""

import matplotlib.pyplot as plt
import numpy as np


with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_7_50h_100epoch/run1/Classical_RBM_7x7_100epoch_50h/loss(gap).txt","r") as file:
    loss_classical_RBM_7_50h_100epoch = [line.strip().replace('\n','') for line in file.readlines()]
    
loss_classical_RBM_7_50h_100epoch=[float(i) for i in loss_classical_RBM_7_50h_100epoch]


with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_7_50h_50epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_7_50h_50epoch = [line.strip().replace('\n','') for line in file.readlines()]
    
loss_classical_RBM_7_50h_50epoch=[float(i) for i in loss_classical_RBM_7_50h_50epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_7_300h_50epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_7_300h_50epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_classical_RBM_7_300h_50epoch=[float(i) for i in loss_classical_RBM_7_300h_50epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_7_300h_100epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_7_300h_100epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_classical_RBM_7_300h_100epoch=[float(i) for i in loss_classical_RBM_7_300h_100epoch]


with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_28_50h_50epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_28_50h_50epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_classical_RBM_28_50h_50epoch=[float(i) for i in loss_classical_RBM_28_50h_50epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_28_50h_100epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_28_50h_100epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_classical_RBM_28_50h_100epoch=[float(i) for i in loss_classical_RBM_28_50h_100epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Classical_RBM_28_300h_10epoch/run1/cost(loss).txt","r") as file:
    loss_classical_RBM_28_300h_10epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_classical_RBM_28_300h_10epoch=[float(i) for i in loss_classical_RBM_28_300h_10epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Quantum_RBM_7_50h_10epoch/run1/cost(loss).txt","r") as file:
    loss_quantum_RBM_7_50h_10epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_quantum_RBM_7_50h_10epoch=[float(i) for i in loss_quantum_RBM_7_50h_10epoch]
    
with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Quantum_RBM_7_50h_50epoch/run1/cost(loss).txt","r") as file:
    loss_quantum_RBM_7_50h_50epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_quantum_RBM_7_50h_50epoch=[float(i) for i in loss_quantum_RBM_7_50h_50epoch]

with open("C:/Users/Alvar/Downloads/TFM_ebm-torch-master/ebm-torch-master_RBM/Alvaro_Project/SubirGitHub/Quantum_RBM_7_50h_100epoch/run1/cost(loss).txt","r") as file:
    loss_quantum_RBM_7_50h_100epoch = [line.strip().replace('\n','') for line in file.readlines()]

loss_quantum_RBM_7_50h_100epoch=[float(i) for i in loss_quantum_RBM_7_50h_100epoch]




x100=np.linspace(1,100,100)
plt.title("RBM gap loss per epoch MNIST 7x7")
plt.xlabel("Training epoch")
plt.ylabel("Free energy Gap")
plt.plot(x100, loss_classical_RBM_7_300h_100epoch, 'r--o', linewidth=0.8, markersize=0.05, label="CRBM_300h")
plt.plot(x100, loss_quantum_RBM_7_50h_100epoch, 'b--*', linewidth=0.8, markersize=0.05, label="HRBM_50h")
plt.plot(x100, loss_classical_RBM_7_50h_100epoch, 'g--o', linewidth=0.8, markersize=0.05, label="CRBM_50h")
#Á: 7x7 plots have the same gap no matter if they are quantum or classical
#plt.plot(x100, loss_classical_RBM_28_50h_100epoch, 'y--o', markersize=0.05)

#plt.legend(["Resnet18 28x28", "Resnet 18 7x7"])
#plt.locator_params(axis="y", nbins=10)

leg=plt.legend(prop={'size': 8})
leg.get_lines()[0].set_linewidth(3)
leg.get_lines()[1].set_linewidth(3)
leg.get_lines()[2].set_linewidth(3)

plt.grid()


plt.savefig("RBM_loss_per_epoch.png")



#Á: Accuracies

x_epochs = [1,10,50,100]

dy_28_50h = [0.98,6.37,1.6,2.71]
acc_28_50h = [1.6,24.8,65.6,66.4]

dy_7_50h=[0.0,1.6,1.79,2.33]
acc_7_50h=[0,9.6,12,16.8]

dy_7_300h=[0,0.8,1.6,3.71]
acc_7_300h=[0,7.2,18.4,26.4]

dy_7_50h_Q=[0,0.98,3.49,3.44]
acc_7_50h_Q=[0,10.4,26.4,39.2]


plt.errorbar(x_epochs, acc_28_50h, yerr=dy_28_50h, fmt='.k', color="yellow", linewidth=0.8)
plt.errorbar(x_epochs, acc_7_300h, yerr=dy_7_300h, fmt='.k', color="red", linewidth=0.8)
plt.errorbar(x_epochs, acc_7_50h_Q, yerr=dy_7_50h_Q, fmt='.k', color="blue", linewidth=0.8)
plt.errorbar(x_epochs, acc_7_50h, yerr=dy_7_50h, fmt='.k', color="green", linewidth=0.8)


plt.title("Accuracies of CRBM and HRBM")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")

plt.plot(x_epochs, acc_28_50h, 'y--o', linewidth=0.8, markersize=0.05, label="CRBM_28_50h")
plt.plot(x_epochs, acc_7_300h, 'r--o', linewidth=0.8, markersize=0.05, label="CRBM_7_300h")
plt.plot(x_epochs, acc_7_50h_Q, 'b--o', linewidth=0.8, markersize=0.05, label="HRBM_7_50h")
plt.plot(x_epochs, acc_7_50h, 'g--o', linewidth=0.8, markersize=0.05, label="CRBM_7_50h")

leg=plt.legend(prop={'size': 8})
leg.get_lines()[0].set_linewidth(3)
leg.get_lines()[1].set_linewidth(3)
leg.get_lines()[2].set_linewidth(3)
leg.get_lines()[3].set_linewidth(3)

plt.grid()
plt.savefig("Grid_RBM_accuracies.png")


#Á: deltav, deltah, deltaw, vbias, hbias, w, v_update, h_update, w_update


plot_deltav_mean=[]
plot_deltah_mean=[]
plot_deltaw_mean=[]

for i in range(len(plot_deltav)):
    plot_deltav_mean.append(plot_deltav[i].mean(0).item())
    plot_deltah_mean.append(plot_deltah[i].mean(0).item())
    plot_deltaw_mean.append(plot_deltaW[i].mean(0).mean(0).item())
    
x_delta=np.linspace(1,len(plot_deltav),len(plot_deltav))
plt.title("Gradient of the loss function")
plt.xlabel("Batch cicle")
plt.ylabel("Gradient of L")
plt.plot(x_delta, plot_deltav_mean, 'r--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$\frac{\partial L}{\partial{b_i}}$")
plt.plot(x_delta, plot_deltaw_mean, 'g--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$\frac{\partial L}{\partial{w_{ij}}}$")
plt.plot(x_delta, plot_deltah_mean, 'b--*', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$\frac{\partial L}{\partial{c_j}}$")
plt.legend(ncol=3)
plt.savefig("Loss_gradients_25epochs_CRBM_7_50h.png")


plot_vbias_mean=[]
plot_hbias_mean=[]
plot_w_mean=[]

for i in range(len(plot_vbias)):
    plot_vbias_mean.append(plot_vbias[i].mean(0).item())
    plot_hbias_mean.append(plot_hbias[i].mean(0).item())
    plot_w_mean.append(plot_w[i].mean(0).mean(0).item())
    
x_bias=np.linspace(1,len(plot_vbias), len(plot_vbias))
plt.title("Biases and weights evolution")
plt.xlabel("Batch cicle")
plt.ylabel("Biases and weights")
plt.plot(x_bias, plot_vbias_mean, 'r--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$b_i$")
plt.plot(x_bias, plot_w_mean, 'g--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$w_{ij}$")
plt.plot(x_bias, plot_hbias_mean, 'b--*', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$c_j$")
plt.legend(ncol=3)
plt.savefig("Biases_and_weights_25epochs_CRBM_7_50h.png")


plot_vbias_update_mean=[]
plot_hbias_update_mean=[]
plot_w_update_mean=[]

for i in range(len(plot_vbias)):
    plot_vbias_update_mean.append(plot_vbias_update[i].mean(0).item())
    plot_hbias_update_mean.append(plot_hbias_update[i].mean(0).item())
    plot_w_update_mean.append(plot_w_update[i].mean(0).mean(0).item())


x_bias=np.linspace(1,len(plot_vbias_update), len(plot_vbias_update))
plt.title(r"$b_i$, $c_j$, $w_{ij}$ " "updates evolution")
plt.xlabel("Batch cicle")
plt.ylabel("Bias and weight updates")
plt.plot(x_bias, plot_vbias_update_mean, 'r--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$b_i$")
plt.plot(x_bias, plot_w_update_mean, 'g--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$w_{ij}$")
plt.plot(x_bias, plot_hbias_update_mean, 'b--*', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$c_j$")
leg=plt.legend(ncol=3)
leg.get_lines()[0].set_linewidth(6)
leg.get_lines()[1].set_linewidth(6)
leg.get_lines()[2].set_linewidth(6)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,4))
plt.grid()
plt.savefig("Grid_bias_and_weight_updates_5epochs_CRBM_7_50h.png")

gradient_vbias0=[]
for i in range(len(plot_deltav)):
    gradient_vbias0.append(plot_deltav[i][0].item())

gradient_w00=[]
for i in range(len(plot_deltav)):
    gradient_w00.append(plot_deltaw[i][0][0].item())

x_w=np.linspace(1,len(plot_deltaw), len(plot_deltaw))
plt.title(r"$\frac{\partial L}{\partial b_i}$ and $\frac{\partial L}{\partial w_{ij}}$" " of " r"$v_0$ and $w_{00}$")
plt.xlabel("Batch cicle")
plt.ylabel("Gradient " r"$\frac{\partial L}{\theta}$", labelpad=-5)
plt.ylim(-9e-06,1e-06)
plt.plot(x_w, gradient_vbias0, 'r--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$\frac{\partial L}{\partial b_i}$")
plt.plot(x_w, gradient_w00, 'b--o', linewidth=0.5, markersize=0.05, alpha=0.4, label=r"$\frac{\partial L}{\partial w_{ij}}$")
leg=plt.legend()
leg.get_lines()[0].set_linewidth(6)
leg.get_lines()[1].set_linewidth(6)
plt.grid()
plt.savefig("Grid_b0_and_w00_gradients_5epochs_CRBM_7_50h.png")

#Á: Annealing sampling

plt.title("Energy histogram")
plt.xlabel("Energy value")
plt.ylabel("Number of occurrences")
ising_df.iloc[:]["energy"].hist(grid=True, bins=150, legend=True)
plt.legend()
plt.grid("True")
plt.savefig("Grid_Energy_Sampling_QA_5epochs_CRBM_7_50h.png")

