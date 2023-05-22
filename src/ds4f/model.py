"""
MIT License
Copyright (c) 2021 HP-SCDS / Observatorio / Máster Data-Science UC /
Diego García Saiz / Jesús González Álvarez / Javier Alejandro Cuartas 
Micieces / 2021-2022 / DeepScan4Failure

This module provides a class which represents the Autoencoder model which must be used so that the main tool class (DeepScanner), works properly.

Functions:

    encoder():  Auxiliar method to ´forward´ which runs all 
                operations and activation functions required, 
                up to latent space layers.
                           
    decoder():  Auxiliar method to ´forward´ which runs all 
                operations and activation functions required, 
                from latent space sampling to output layer.
                           
    forward():  Common method in torch models, which executes 
                all operations and activation functions, and It 
                aggregates the result of the loop (a ´decoder´ 
                output per sample from latent space) as a single 
                pair of posterior parameters by running the mean.
                           
    loss_fn():  This method runs the ELBO as the function to be 
                optimized, assuming as the prior, a Normal(0,I) 
                of mean 0 and Identity matrix for the covariance 
                matrix for the Kullback Leiber divergence term, 
                and assuming the log likelihood of the posterior 
                as the reconstruction loss term.
    
Classes:

    Autoencoder: allows loading data from HDF-5 input files.
    
"""

import torch
import torch.nn.functional as F

class Autoencoder(torch.nn.Module):
    """
    Model class.
    Child class of torch nn.Module class with network architecture, loss function, and forward methods as usual in torch models. A variable number of layers was preferred as an hyperparameter to control the complexity of the Autoencoder network, so instead of torch.nn.Sequential in a common python list, as [1] suggests, torch.nn.ModuleList was used. This way existing problems when tracing neural network parameters are avoided, while training and performing inference. Linear and BatchNorm1d layers were used for the network architecture along with uniform_ random initialization. ReLu and Sigmoid activation function were used in ´forward´ method.
    
    Attributes:
    
      inputsz         (int): Input size of the network.
                    
      encodersz      (list): List of sizes of the network, hidden 
                             sizes from the input size (not included) 
                             to the latent space size (included). It 
                             is reversed to get the symmetric part of 
                             the architecture as autoencoder.
                  
      L               (int): Total number of samples taken from the 
                             latent space whenever the forward method 
                             is called, to feed the ´decoder´ method.
      layerlst  
      (torch.nn.ModuleList): Object including the architecture of the 
                             neural network (but activation functions, 
                             considered in ´forward´ method and its 
                             auxiliar methods ´encoder´ and ´decoder´.
    Methods:
    
      encoder():             Auxiliar method to ´forward´ which runs all 
                             operations and activation functions required,
                             up to latent space layers.
                             
      decoder():             Auxiliar method to ´forward´ which runs all 
                             operations and activation functions required, 
                             from latent space sampling to output layer.
                             
      forward():             Common method in torch models, which executes 
                             all operations and activation functions, and It 
                             aggregates the result of the loop (a ´decoder´ 
                             output per sample from latent space) as a single 
                             pair of posterior parameters by running the mean.
                             
      loss_fn():             This method runs the ELBO as the function to be 
                             optimized, assuming as the prior, a Normal(0,I) 
                             of mean 0 and Identity matrix for the covariance
                             matrix for the Kullback Leiber divergence term, 
                             and assuming the log likelihood of the posterior 
                             as the reconstruction loss term.
      
    [1]https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/2
    """
      
    def __init__(self, input_size=1,hidden_sizes=[],z_samplesz=5):
      """
      Torch Model's Constructor Class.
      
      Kwargs:
      
        input_size      (int):   Input size of the autoencoder function.
                    
        hidden_sizes   (list):   List of sizes of the network, hidden sizes 
                                 from the input size (not included) to the 
                                 latent space size (included).
                    
        z_samplesz      (int):   Number of samples taken from the latent space 
                                 each ´forward´ call.
                   
      [1]https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/2
      """
    
      super().__init__()
      
      self.inputsz=input_size
      self.encodersz=hidden_sizes.copy()
      self.L=z_samplesz
      
      # layerlst is a torch ModuleList built with all encoder and decoder 
      # layers, to avoid widely known problems regarding parameter tracing 
      # when using common lists and sequential models.
      
      self.layerlst=torch.nn.ModuleList()

      for i in range(len(hidden_sizes)):

          # Input and output size of encoder layers are read.
          
          if i==0:
              inputs=input_size
              outputs=hidden_sizes[i]
          else:
              inputs=hidden_sizes[i-1]
              outputs=hidden_sizes[i]
          
          # Encoder layers are appended
          
          self.layerlst.append(torch.nn.Linear(
              in_features=int(inputs), out_features=int(outputs)
          ))
          torch.nn.init.uniform_(self.layerlst[len(self.layerlst)-1].weight,a=0,b=1)

          # Latent Space additional layer is appended here (for variance
          # and mean):
          
          if i==(len(hidden_sizes)-1):
              
            self.layerlst.append(torch.nn.BatchNorm1d(
                int(outputs)
            ))
            self.layerlst.append(torch.nn.Linear(
                in_features=int(inputs), out_features=int(outputs)
            ))
                
            torch.nn.init.uniform_(self.layerlst[len(self.layerlst)-1].weight,a=0,b=1)

            self.layerlst.append(torch.nn.BatchNorm1d(
                int(outputs)
            ))
                
          else:
            self.layerlst.append(torch.nn.BatchNorm1d(
                int(outputs)
            ))
            
      # Hidden layer sizes input list is reversed so as to append decoder
      # layers.
      
      inv_hidden_sizes=hidden_sizes.copy()
      inv_hidden_sizes.reverse()

      for i in range(len(inv_hidden_sizes)):

          # Input and output size of decoder layers are read.
          
          if i<(len(inv_hidden_sizes)-1):
              inputs=inv_hidden_sizes[i]
              outputs=inv_hidden_sizes[i+1]
          else:
              inputs=inv_hidden_sizes[i]
              outputs=input_size
              
          # Decoder layers are appended
          
          self.layerlst.append(torch.nn.Linear(
              in_features=int(inputs), out_features=int(outputs)
          ))
          torch.nn.init.uniform_(self.layerlst[len(self.layerlst)-1].weight,a=0,b=1)

          if i==(len(inv_hidden_sizes)-1):
            self.layerlst.append(torch.nn.Linear(
                in_features=int(inputs), out_features=int(outputs)
            ))
            torch.nn.init.uniform_(self.layerlst[len(self.layerlst)-1].weight,a=0,b=1)

          else:
            self.layerlst.append(torch.nn.BatchNorm1d(
                int(outputs)
            ))

    def encoder(self,x):
      """
      Auxiliar to ´forward´ method. It runs all operations and activation functions required, up to latent space layers when called. ReLu activation function is applied to the output of all Normalization layers considered.

      Args:
      
        x      (torch.tensor): Input tensor coming from the ´forward´ method 
                               previous steps.
        
      Returns:
      
        torch.tensor input ´x´ is processed following the architecture defined 
        in model, and the output of the method is the same format and name as 
        the input.
      
      """
        
      il=0
      for layer in self.layerlst[:-(len(self.encodersz)*2-1+5)]:
          if all([(il+1)%2==0]):
              x = F.leaky_relu(layer(x),negative_slope=0.01)
          else:
              x=layer(x)
          il=il+1
      return x
        
    def decoder(self,x,mu_lat,logv_lat):
      """                      
      Auxiliar to ´forward´ method. It runs all operations and activation functions required, from latent space sampling to output layers when called. It starts with the reparametrization trick and then, ReLu activation function is applied to the output of all Normalization layers considered. Sigmoid activation is applied to the output.
      
      Args:
      
        x        (torch.tensor): Input tensor coming from the ´forward´ method 
                                 from previous steps. 
        
        mu_lat   (torch.tensor): Mean parameter for the Normal function 
                                 established as prior.
        
        logv_lat (torch.tensor): Logarithm of the variance (set as such because 
                                 of convergence and computation benefits), 
                                 parameter for the Normal function established 
                                 as prior.
      Returns:
      
        torch.tensors res_mu_li,res_logv_li are the posterior parameters, mean 
        and variance(logarithm of the variance for convergence and computation 
        benefits).
      
      """
      x = torch.randn(x.shape[0],self.encodersz[len(self.encodersz)-1])*torch.exp(0.5 * logv_lat)+mu_lat
      ily=0
      for layer in self.layerlst[(len(self.encodersz)*2-1+3):-2]:
          if ily==0:
              w=layer(x)
          elif (ily+1)%2==0:
              w = F.leaky_relu(layer(w),negative_slope=0.01)
          else:
              w = layer(w)
          ily=ily+1
              
      res_mu_li=torch.sigmoid(self.layerlst[-2](w))
      res_logv_li=torch.sigmoid(self.layerlst[-1](w))
      
      return res_mu_li,res_logv_li
        
    def forward(self, x):
      """
      Common method in torch models, which executes all operations and activation functions required, and It aggregates the result of the loop (a ´decoder´ output per sample from latent space) as a single pair of posterior parametersby running the mean.
          
      Args:
      
        x        (torch.tensor):  Input tensor.
        
      Returns:
      
        mu_lat   (torch.tensor):  Mean parameter for the posterior.
        
        torch.exp(0.5 * logv_res)        
                 (torch.tensor):  Variance parameter for the posterior.
                  
        mu_lat   (torch.tensor):  Mean parameter for the Normal function established 
                                  as prior.
        
        logv_lat (torch.tensor):  Logarithm of the variance (set as such because of 
                                  convergence and computation benefits), parameter 
                                  for the Normal function established as prior.        
      """
     
      # Encoder is applied up to the latent space layers
      
      x=self.encoder(x)
      
      # Latent space linear combination output is Normalized but no ReLu is applied
      
      xm=self.layerlst[(len(self.encodersz)*2-1-1)](x)
      mu_lat = self.layerlst[(len(self.encodersz)*2-1)](xm)
      
      xv=self.layerlst[len(self.encodersz)*2](x)
      logv_lat = self.layerlst[len(self.encodersz)*2+1](xv)
      
      res_mu_l=[]
      res_logv_l=[]
      
      # Latent space is sampled with size self.L, and decoder is applied.
      
      for l in range(self.L):
          res_mu_li,res_logv_li=self.decoder(x,mu_lat,logv_lat)
          res_mu_l.append(res_mu_li)
          res_logv_l.append(res_logv_li)
          
      # Decoder output is stacked and transformed to get the posterior parameters.
      
      mu_res = torch.mean(torch.stack(res_mu_l),dim=0)
      logv_res = torch.mean(torch.stack(res_logv_l),dim=0)
      
      return [mu_res,torch.exp(0.5 * logv_res), mu_lat, logv_lat]       

    def loss_fn(self,output,inputs,beta,tset):
      """
      This method implements the ELBO as the function to be optimized, assuming the prior is a Normal(0,I) of mean 0 and Identity matrix for the covariance matrix, with the proper Kullback Leiber divergence term as regularizer, and the log likelihood of the posterior as the reconstruction loss term.
      
      Args:
      
        output           (list):  Output list of tensors from ´forward´ method's 
                                  execution on a given input.
        
        inputs            (int):  Input torch tensor to be compared against ouput 
                                  parameters.
        
        beta              (int):  Parameter for the Kullback Leiber term, to set 
                                  its importance in optimization.
        
        tset              (str):  This argument should be set to "t" when calling 
                                  loss_fn during training, to compute the mean
                                  for each batch elements.
        
      Returns:
      
        Tensor with a single aggregated value when tset="t" and a value per batch 
        input element when tset="t".
      
      [2] beta-VAE: Higgins I, Matthey L, Pal A et al. Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2017.
      """
      f=-0.5 *torch.sum(1 + output[3] - output[2]**2 - output[3].exp(),dim=1)
      d=torch.distributions.Normal(output[0],output[1]).log_prob(inputs)
      
      if tset=="t":
      #  When loss_fn is used for training, mean of the loss_fn for all the batch elements is calculated.
        d=-torch.sum(d,dim=1)
        f=f
        res=torch.mean(d+beta*f)
      else:
      # When loss_fn is used for inference, each loss_fn output for each batch element is calculated.
        d=-torch.sum(d,dim=1)
        f=f
        res=d+beta*f
        
      return res
    