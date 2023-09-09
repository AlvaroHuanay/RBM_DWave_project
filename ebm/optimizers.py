# Collection of optimizers for RBMs in pytorch
#
# Author: Alejandro Pozas-Kerstjens
# Edited: Álvaro Huanay de Dios
# Requires: pytorch as ML framework
# Last modified: Jul, 2019

import torch
from torch.nn.functional import linear
from ebm.quantum_annealing import quantum_annealing
import timeit
from timeit import default_timer as timer


def outer_product(vecs1, vecs2):
    '''Computes the outer product of batches of vectors

    Arguments:

        :param vecs1: b 1-D tensors of length m
        :type vecs1: list of torch.Tensor
        :param vecs2: b 1-D tensors of length n
        :type vecs2: list of torch.Tensor
        :returns: torch.Tensor of size (m, n)
       '''
    return torch.bmm(vecs1.unsqueeze(2), vecs2.unsqueeze(1))


class Optimizer(object):

    def __init__(self):
        '''Constructor for the class.
        '''

    def get_updates(self, vpos, vneg, weights, vbias, hbias):
        '''Obtains the parameter updates for weights and biases

        Arguments:

            :param vpos: Positive phase of the visible nodes
            :type vpos: torch.Tensor
            :param vneg: Negative phase of the visible nodes
            :type vneg: torch.Tensor
            :param weights: Weights connecting the visible and hidden layers
            :type weights: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter

            :returns: :W_update: torch.Tensor -- update for the RBM weights
                      :vbias_update:  torch.Tensor -- update for the RBM
                                      visible biases
                      :hbias_update:  torch.Tensor -- update for the RBM
                                      hidden biases
        '''

class SGD(Optimizer):

    def __init__(self, learning_rate, momentum=0, weight_decay=0):
        '''Constructor for the class.

        Arguments:

            :param learning_rate: Learning rate
            :type learning_rate: float
            :param weight_decay: Weight decay parameter, to prevent overfitting
            :type weight_decay: float
            :param momentum: Momentum parameter, for improved learning
            :type momentum: float
        '''
        super(SGD, self).__init__()
        assert learning_rate >= 0, ('You should specify ' +
                                    'a valid learning rate (>=0), given is ' +
                                    str(learning_rate))
        self.learning_rate = learning_rate
        assert momentum >= 0, ('You should specify a valid momentum (>=0), ' +
                               'given is ' + str(momentum))
        self.momentum = momentum
        assert weight_decay >= 0, ('You should specify ' +
                                   'a valid weight decay (>=0), given is' +
                                   str(weight_decay))
        self.weight_decay = weight_decay
        self.first_call = True
        self.epoch = 0

    def get_updates(self, vpos, vneg, weights, vbias, hbias):
        if self.first_call:
            self.W_update = torch.zeros_like(weights)
            self.vbias_update = torch.zeros_like(vbias)
            self.hbias_update = torch.zeros_like(hbias)
            self.first_call = False

        self.W_update     *= self.momentum
        self.hbias_update *= self.momentum
        self.vbias_update *= self.momentum

        # Weight decay is only applied to weights, because they are the maximum
        # responsibles for overfitting
        # Note that we multiply by the learning rate, so the function
        # optimized is (NLL - weight_decay * weights)
        self.W_update -= self.learning_rate * self.weight_decay * weights

        hpos = torch.sigmoid(linear(vpos, weights, hbias))
        hneg = torch.sigmoid(linear(vneg, weights, hbias))
        deltaW = (outer_product(hpos, vpos).mean(0)
                  - outer_product(hneg, vneg).mean(0))
        deltah = hpos.mean(0) - hneg.mean(0)
        deltav = vpos.mean(0) - vneg.mean(0)

        self.W_update.data     += self.learning_rate * deltaW
        self.vbias_update.data += self.learning_rate * deltav
        self.hbias_update.data += self.learning_rate * deltah

        return self.W_update, self.vbias_update, self.hbias_update

class Adam(Optimizer):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        '''Constructor for the class.

        Arguments:

            :param learning_rate: Learning rate
            :type learning_rate: float
            :param beta1: Adam parameter, for first moment regularization
            :type beta1: float
            :param beta2: Adam parameter, for second moment regularization
            :type beta2: float
            :param eps: Adam parameter, to prevent divergences
            :type eps: float
        '''
        super(Adam, self).__init__()
        assert learning_rate >= 0, ('You should specify ' +
                                    'a valid learning rate (>=0)')
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.first_call = True
        self.epoch = 0

    def get_updates(self, vpos, vneg, weights, vbias, hbias, rbm_state_dict, batch_num, classical, _ , p, epochs, qubo, sampler_auto):
        
        if self.first_call:
            self.m_W = torch.zeros_like(weights)
            self.m_v = torch.zeros_like(vbias)
            self.m_h = torch.zeros_like(hbias)
            self.v_W = torch.zeros_like(weights)
            self.v_v = torch.zeros_like(vbias)
            self.v_h = torch.zeros_like(hbias)
            self.first_call = False
        
        
        if (classical==False and _==epochs-1 and p>=59000):
            #p: index of the image. The total is 60.000, leave the restant 3800 images to train with QA
            #1 entire epoch requires 13 hours running that´s why only the last part is run by DWave
            hpos = torch.sigmoid(linear(vpos, weights, hbias))
            print("Intro quantum annealing...\n")
            start_qa_process = timer()
            vneg, hneg, wneg = quantum_annealing(rbm_state_dict, batch_num, qubo, sampler_auto)
            end_qa_process = timer()
            elapse=end_qa_process - start_qa_process
            print("Time running quantum annealing process: \n", elapse)
            
            deltah = hpos.mean(0) - hneg #Á: No mean porque las dimensiones ya están preparadas para que haga el cálculo
            deltav = vpos.mean(0) - vneg #Á: Lo mismo
            deltaW = (outer_product(hpos, vpos).mean(0)
                      - wneg)            #Á: Lo mismo 

        

        #if (classical==False and _==epochs-1 and p>=59000):
            #p: index of the image. The total is 60.000, leave the restant 3800 images to train with QA
            #1 entire epoch requires 13 hours running that´s why only the last part is run by DWave
            #print("Intro quantum annealing...\n")
            #start_qa_process = timer()
            #vneg, hneg, wneg = quantum_annealing(rbm_state_dict, batch_num, qubo, sampler_auto)
            #vneg = quantum_annealing(rbm_state_dict, batch_num, qubo, sampler_auto)
            #end_qa_process = timer()
            #elapse=end_qa_process - start_qa_process
            #print("Time running quantum annealing process: \n", elapse)
            
            #hneg = torch.sigmoid(linear(vneg, weights, hbias))
            #hpos = torch.sigmoid(linear(vpos, weights, hbias))
            #deltah = hpos.mean(0) - hneg.mean(0) #Á: No mean porque las dimensiones ya están preparadas para que haga el cálculo
            #deltav = vpos.mean(0) - vneg #Á: Lo mismo
            #w=torch.zeros(hneg.size(0),vneg.size(0))
            #for j in range(hneg.size(0)):
            #    for i in range(vneg.size(0)):
            #        w[j][i]=hneg[j]*vneg[i]
            #print(hneg.size())
            #print(vneg.size())
            #print(hpos.size())
            #print(vpos.size())
            #print(outer_product(hpos, vpos).mean(0).size())
            #print(outer_product(hpos, vpos).size())
            #print(w.size())
            #deltaW = (outer_product(hpos, vpos).mean(0)
            #          - w)           #Á: Lo mismo

        else:
            hneg = torch.sigmoid(linear(vneg, weights, hbias))
            #Á doubt: Es esto negative phase de hidden bias?
            #print(hneg.size()) #torch.Size([100, 300])
            hpos = torch.sigmoid(linear(vpos, weights, hbias))
            #print(hpos.size()) #torch.Size([32, 300])
            
            
            deltah = hpos.mean(0) - hneg.mean(0)
            deltav = vpos.mean(0) - vneg.mean(0)
            deltaW = (outer_product(hpos, vpos).mean(0)
                      - outer_product(hneg, vneg).mean(0))
        


        #Á doubt: Es outer_product(hneg, vneg).mean(0) negative phase de weight?
        #Á doubt: Es vneg negative phase de visible bias?
        """
        Aquí es donde tendría que usar los resultados del annealing
        <>_{model} para calcular los deltas de las loss function y así los
        updates de cada iteración de las X totales dentro de cada epoch.
        
        vpos.size() #torch.size([32,49])
        vneg.size() #torch.size([100,49])
        hpos.size() #torch.Size([32, 300])
        hneg.size() #torch.Size([100, 300]
        wpos.size() #torch.Size([300, 49]) (outer_product(hpos, vpos).mean(0))
        wneg.size() #torch.Size([300, 49]) (outer_product(hneg, vneg).mean(0))
        
        Importante!
        
        vpos.mean(0) #torch.Size([49])
        hpos.mean(0) #torch.Size([300])
        wpos.mean(0) #torch.Size([300, 49])
        
        vneg.mean(0) #torch.Size([49]) (Á annealing: #torch.Size([49,1]))
        hneg.mean(0) #torch.Size([300]) (Á annealing: #torch.Size([300,1]))
        wneg.mean(0) #torch.Size([300, 49]) (Á annealing: torch.Size([300, 49]))

        
        """


        self.m_W *= self.beta1
        self.m_W += (1 - self.beta1) * deltaW
        self.m_v *= self.beta1
        self.m_v += (1 - self.beta1) * deltav
        self.m_h *= self.beta1
        self.m_h += (1 - self.beta1) * deltah

        self.v_W *= self.beta2
        self.v_W += (1 - self.beta2) * deltaW * deltaW
        self.v_v *= self.beta2
        self.v_v += (1 - self.beta2) * deltav * deltav
        self.v_h *= self.beta2
        self.v_h += (1 - self.beta2) * deltah * deltah

        mnorm_W = self.m_W / (1 - self.beta1 ** (self.epoch + 1))
        mnorm_v = self.m_v / (1 - self.beta1 ** (self.epoch + 1))
        mnorm_h = self.m_h / (1 - self.beta1 ** (self.epoch + 1))

        vnorm_W = self.v_W / (1 - self.beta2 ** (self.epoch + 1))
        vnorm_v = self.v_v / (1 - self.beta2 ** (self.epoch + 1))
        vnorm_h = self.v_h / (1 - self.beta2 ** (self.epoch + 1))

        W_update     = self.learning_rate * mnorm_W / (torch.sqrt(vnorm_W) + self.eps)
        vbias_update = self.learning_rate * mnorm_v / (torch.sqrt(vnorm_v) + self.eps)
        hbias_update = self.learning_rate * mnorm_h / (torch.sqrt(vnorm_h) + self.eps)
        

        return W_update, vbias_update, hbias_update, deltah, deltav, deltaW

class AdaBound(Optimizer):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8,
                 final_lr=0.1, gamma=1e-3):
        '''AdaBound optimizer from
        https://www.luolc.com/publications/adabound/

        Arguments:

            :param learning_rate: Learning rate
            :type learning_rate: float
            :param beta1: Adam parameter, for first moment regularization
            :type beta1: float
            :param beta2: Adam parameter, for second moment regularization
            :type beta2: float
            :param eps: Adam parameter, to prevent divergences
            :type eps: float
            :param final_lr: learning rate to which the process should converge.
                             The method should be robust to this parameter
            :type final_lr: float
            :param gamma: parameter on the bounds function. Method should be
                          robust to this parameter
            :type gamma: float
        '''
        super(AdaBound, self).__init__()
        assert learning_rate >= 0, ('You should specify ' +
                                    'a valid learning rate (>=0)')
        self.learning_rate = learning_rate
        self.beta1         = beta1
        self.beta2         = beta2
        self.eps           = eps
        self.final_lr      = final_lr
        self.gamma         = gamma
        self.first_call    = True
        self.epoch         = 0
        self.initial_lr    = learning_rate

    def get_updates(self, vpos, vneg, weights, vbias, hbias):
        if self.first_call:
            self.m_W = torch.zeros_like(weights)
            self.m_v = torch.zeros_like(vbias)
            self.m_h = torch.zeros_like(hbias)
            self.v_W = torch.zeros_like(weights)
            self.v_v = torch.zeros_like(vbias)
            self.v_h = torch.zeros_like(hbias)
            self.first_call = False

        hpos = torch.sigmoid(linear(vpos, weights, hbias))
        hneg = torch.sigmoid(linear(vneg, weights, hbias))
        deltaW = (outer_product(hpos, vpos).mean(0)
                  - outer_product(hneg, vneg).mean(0))
        deltah = hpos.mean(0) - hneg.mean(0)
        deltav = vpos.mean(0) - vneg.mean(0)

        self.m_W *= self.beta1
        self.m_W += (1 - self.beta1) * deltaW
        self.m_v *= self.beta1
        self.m_v += (1 - self.beta1) * deltav
        self.m_h *= self.beta1
        self.m_h += (1 - self.beta1) * deltah

        self.v_W *= self.beta2
        self.v_W += (1 - self.beta2) * deltaW * deltaW
        self.v_v *= self.beta2
        self.v_v += (1 - self.beta2) * deltav * deltav
        self.v_h *= self.beta2
        self.v_h += (1 - self.beta2) * deltah * deltah

        m_correction = 1 - self.beta1 ** (self.epoch + 1)
        v_correction = 1 - self.beta2 ** (self.epoch + 1)
        step = self.learning_rate * np.sqrt(v_correction) / m_correction
        
        lr = self.final_lr * self.learning_rate / self.initial_lr
        upper_bound = lr * (1 + 1 / (self.gamma * (self.epoch + 1) + 1))
        lower_bound = lr * (1 - 1 / (self.gamma * (self.epoch + 1)))
        
        W_update     = (step / (torch.sqrt(self.v_W) + self.eps)).clamp_(lower_bound, upper_bound) * self.m_W
        vbias_update = (step / (torch.sqrt(self.v_v) + self.eps)).clamp_(lower_bound, upper_bound) * self.m_v
        hbias_update = (step / (torch.sqrt(self.v_h) + self.eps)).clamp_(lower_bound, upper_bound) * self.m_h

        return W_update, vbias_update, hbias_update
