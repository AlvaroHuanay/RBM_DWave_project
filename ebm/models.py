# Collection of energy-based ML models
#
# Author: Alejandro Pozas-Kerstjens
# Edited: Álvaro Huanay de Dios
# Requires: copy for variable copies
#           numpy for numerics
#           pytorch as ML framework
#           tqdm for progress bar
# Last modified: Jul, 2019

import torch
from copy import deepcopy
from .optimizers import outer_product
from torch.nn import Module, Parameter
from torch.nn.functional import linear, softplus
from tqdm import tqdm
import numpy as np


class RBM(Module):

    def __init__(self, n_visible=100, n_hidden=50, sampler=None, optimizer=None,
                 device=None, weights=None, hbias=None, vbias=None):
        '''Constructor for the class.

        Arguments:

            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param verbose: Optional parameter to set verbosity mode
            :type verbose: int
            :param weights: Optional parameter to specify the weights of the RBM
            :type weights: torch.nn.Parameter
            :param hbias: Optional parameter to specify the hidden biases of
                          the RBM
            :type hbias: torch.nn.Parameter
            :param vbias: Optional parameter to specify the visibile biases of
                          the RBM
            :type vbias: torch.nn.Parameter
        '''

        super(RBM, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')

        if weights is not None:
            self.weights = Parameter(weights.to(self.device))
        else:
            self.weights = Parameter(0.01 * torch.randn(n_hidden,
                                                        n_visible
                                                        ).to(self.device))

        if hbias is not None:
            self.hbias = Parameter(hbias.to(self.device))
        else:
            self.hbias = Parameter(torch.zeros(n_hidden).to(self.device))

        if vbias is not None:
            self.vbias = Parameter(vbias.to(self.device))
        else:
            self.vbias = Parameter(torch.zeros(n_visible).to(self.device))
        
        for param in self.parameters():
            param.requires_grad = False

        if optimizer is None:
            raise Exception('You must provide an appropriate optimizer')
        self.optimizer = optimizer

        if sampler is None:
            raise Exception('You must provide an appropriate sampler')
        self.sampler = sampler

    def free_energy(self, v):
        '''Computes the free energy for a given state of the visible layer.

        Arguments:

            :param v: The state of the visible layer of the RBM
            :type v: torch.Tensor

            :returns: torch.Tensor
        '''
        vbias_term = v.mv(self.vbias)
        wx_b = linear(v, self.weights, self.hbias)
        hidden_term = softplus(wx_b).sum(1)
        return (-hidden_term - vbias_term)

    def train(self, input_data, rbm_state_dict, classical,_, epochs, qubo, sampler_auto, plot_weights, plot_vbias, plot_hbias, plot_deltah, plot_deltav, plot_deltaw, plot_vbias_update, plot_hbias_update, plot_w_update):
        '''Trains the RBM.

        Arguments:

            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
        '''
        #elapse_list=[]
        batch_num=0
        p=0
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):
            sample_data = batch.float()
            p=p+len(batch)
            #batch_num=batch_num+1
            #print(len(input_data))
            # Sampling from the model to compute updates
            # Get positive phase from the data
            vpos = sample_data #torch.size([32,49])
            # Get negative phase from the chains
            vneg = self.sampler.get_negative_sample(vpos, self.weights,
                                                    self.vbias, self.hbias)
            #torch.size([100,49])
            #Á: Es que necesita el primer sampling del parallel para arrancar
            #y luego todo quantum annealing para optimizar? Entonces sería
            # centrarnos en optimizer.get_updates en vez de
            #sampler.get_negative_sample
            #print(vneg.size())
            """
            print("vneg type: ", type(vneg))
            print("vneg size: ", vneg.size())
            print("vpos type: ", type(vpos))
            print("vpos size: ", vpos.size())
            print("len(vneg[0]) ", len(vneg[0]))
            print("len(vneg) ", len(vneg))
            """
            # Weight updates. Includes momentum and weight decay
            
            W_update, vbias_update, hbias_update, deltah, deltav, deltaW = \
                        self.optimizer.get_updates(vpos, vneg, self.weights,
                                                    self.vbias, self.hbias, rbm_state_dict, batch_num, classical,_, p, epochs, qubo, sampler_auto)
            
            self.weights += W_update
            #print(self.weights.size())
            self.hbias   += hbias_update
            self.vbias   += vbias_update
            
            plot_weights.append(self.weights)
            plot_vbias.append(self.vbias)
            plot_hbias.append(self.hbias)
            
            plot_vbias_update.append(vbias_update)
            plot_hbias_update.append(hbias_update)
            plot_w_update.append(W_update)
            
            plot_deltah.append(deltah)
            plot_deltav.append(deltav)
            plot_deltaw.append(deltaW)
            
            
            #elapse_list.append(elapse)
            batch_num=batch_num+1
            
        self.optimizer.epoch += 1
        #avg_annealing=1/len(elapse_list)*np.cumsum(elapse_list)[-1]
        #print("The avg annealing time process is: ", avg_annealing)
        