"""
This module provides utilities for the main tool class to work properly (DeepScanner).

Functions:

    __len__():     Number of blocks of size ´nfeatures´ 
                   (outputs) this Dataset holds.

    __getitem__(): Method to get items from the Dataset 
                   represented by the class.

    run():         It runs the inference from an input 
                   list with the posterior parameters and 
                   the input tensor which probability the
                   user is trying to infer.
                
    mcc():         It gets the Mathew's Correlation Coefficient 
                   from a numpy array with labels and another 
                   with predictions.
    
    prc():         It gets the Precision from a numpy array 
                   with labels and another with predictions.
    
    rcc():         It gets the Recall from a numpy array with 
                   labels and another with predictions.
    
Classes:

    H5LDataset: allows loading data from HDF-5 input files.
    
    DecisionTaker: allows running predictions from the main tool class.
    
"""

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

class H5LDataset(Dataset):

    """
    Child class of torch Dataset class.

    Child class of torch Dataset class. Original project data was organised in a parquet file, with a time series of a large number of measurements in each column. This class expects an input in .h5 (HDF-5) format since a  preprocessing was performed using this format and It was preferred to `.parquet` taking into account large size of initial data. 

    Each column of ´ncolumns´ is expected to be a set of variables corresponding to a single signal or time series. These will be combined with others in sets of size ´nfeatures´ by the method ´__getitem__´ to become the input of the Autoencoder, so the output size of ´__getitem__´will be each column´s length times 3.

    HDF-5 input file from `file` is expected to have a `general/` group, and index numbers as dataset names. Each dataset is expected to have a single row stored corresponding to a single time series feature, and sets of ´nfeatures´ time series must be identified by consecutive numbers in dataset names.

    Attributes:

        ids  (pd.Series):   Pandas series with column indexes of the
                            parquet file (datasets names in the HDF-5 
                            format file) that can be accessed with 
                            ´__getitem__´ method.
                            
        idssz      (int):   Length of ´ids´.
                    
        df         (str):   Path to HDF-5 data file. Group ´general/´
                            expected. Dataset names must be ordered 
                            numbers. Succesive column indexes must be 
                            grouped in columns so that this Dataset
                            class constructor can split them into 
                            separate observations of nfeature columns.
                    
        nrows      (int):   Total number of variables per signal (column) 
                            of the .h5 file (or observations in each signal 
                            in the original approach).
                    
        ncolumns   (int):   Total number of signals/time series in the 
                            dataset.

        nfeatures  (int):   Number of columns per observation in the dataset. 
                            Each observation is defined by a set of different 
                            features which were measured with the same frequency. 
                            This number of consecutive dataset ids will be 
                            concatenated and flattened as the autoencoder input.
                           
        cinit     (list):   List of inital indexes for all groups of size 
                            ´nfeatures´ which helps ´__getitem__´ to build its 
                            output.
                            
        cfinl     (list):   List of final indexes for all groups of size 
                            ´nfeatures´  which helps ´__getitem__´ to build its 
                            output.
                     
        transform 
        (torch.transform):  Transform method to apply to ´__getitem__´ outputs

    Methods:

        __len__():          Number of blocks of size ´nfeatures´ (number of 
                            outputs), this Dataset holds.
                            
        __getitem__():      It gets a float type tensor output of index ´idx´ 
                            (method parameter) from the dataset, along with an 
                            error flag.

    """

    def __init__(self, ids, file, nrows, ncolumns, nfeatures, transform=None): 
      """
      Torch Dataset Class Constructor from HDF-5 files according to the described structured.
     
      Args:
     
          ids  (pd.Series):   Pandas series with column indexes of the parquet 
                              file (datasets names in the HDF-5 format file) that 
                              can be accessed with ´__getitem__´ method.
                      
          file       (str):   Path to HDF-5 data file. Group ´general/´expected. 
                              Dataset names must be ordered numbers. Succesive 
                              column indexes must be grouped in columns so that 
                              this Dataset class constructor can split them into 
                              separate observations of nfeature columns.
                      
          nrows      (int):   Total number of variables per signal (column) of the 
                              .h5 file (or observations in each signal in the 
                              original approach).
                      
          ncolumns   (int):   Total number of signals/time series in the dataset.
     
          nfeatures  (int):   Number of columns per observation in the dataset. Each 
                              observation is defined by a set of different features 
                              which were measured with the same frequency. This number 
                              of consecutive dataset ids will be concatenated
                              and flattened as the autoencoder input.
                          
      Kwargs:
     
          transform 
          (torch.transform):  Transform method to apply to ´__getitem__´ outputs
     
      Raise:
     
          A ValueError exception will be raised when the number of columns considered 
          in ´ids´ is not multiple of ´nfeatures´.
     
      """
      self.idssz=len(ids)
      if self.idssz%nfeatures==0:
        self.transform = transform
        self.nrows=nrows
        self.nfeatures=nfeatures
        self.df=file
        self.cinit=[ids[n] for n in np.arange(0,self.idssz,self.nfeatures)]
        self.cfinl=[ids[n-1] for n in np.arange(nfeatures,self.idssz+1,self.nfeatures)]
        
      else:
        raise ValueError("Dataframe size is not multiple of the number of features per observation, provided.")

    def __len__(self):
      """
      Number of blocks of size ´nfeatures´ (outputs), this Dataset holds.
    
      Returns:
    
          Integer output of the number of blocks of size ´nfeatures´ the Dataset holds.
    
      """
      return int(self.idssz/self.nfeatures)

    def __getitem__(self, idx):
      """
      Method to get items from the Dataset represented by the class.It uses an index for each group of signals of size ´nfeature´ (single sample) and attributes ´cinit´ and ´cfinl´ to read the proper datasets from group ´general/´ of the HDF-5 file.
  
      Args:
  
          idx        (int):   Index of a single block of signals corresponding to 
                              a single sample (single output).can be accessed with 
                              ´__getitem__´ method.
  
      Returns:
  
          Float type tensor output of index idx from the dataset, along with a non 
          error flag.
  
      """
      with h5py.File(self.df, "r") as f:
        b=list()
        for el in range(self.cinit[idx],self.cfinl[idx]+1,1):
          nx=f["general/"+str(el)][:]
          b=b+nx[:,:].tolist()[0]
        res=np.array(b)
      if self.transform:
        res = self.transform(res)
      return res.float(), 0
      
class DecisionTaker:

    """
    This class includes the probability inference tool and metrics to control for 
    undesired outputs.

    Methods:

        run():      It runs the inference from an input list with the posterior 
                    parameters and the input tensor which probability the user 
                    is trying to infer.
                    
        mcc():      It gets the Mathew's Correlation Coefficient from a numpy 
                    array with labels and another with predictions.

        prc():      It gets the Precision from a numpy array with labels and 
                    another with predictions.

        rcc():      It gets the Recall from a numpy array with labels and another
                    with predictions.

    """

    def run(self,params):
      """
      Inference method to get probabilities of a vector given the trained latent space with the posterior parameters.
  
      Args:
  
          params    (list):   List of 2 terms. The first is expected to contain 
                              at least 2 elements, the parameters of the posterior 
                              in two tensors (output of the ´forward´ method of the 
                              Autoencoder class). The second element of the list is 
                              expected to contain a torch tensor with the input vector.
  
      Returns:
  
          Torch tensor with the mean of the probabilities from the posterior.
  
      """
      result=torch.mean(torch.distributions.Normal(params[0][0],params[0][1]).log_prob(params[1]).exp(),dim=1)
      return result

    def mcc(self,labels,outputs):
      """
      It gets the Mathew's Correlation Coefficient (MCC) from a numpy array with labels and another with predictions.
  
      Args:
  
          labels  (pandas.Series):   Labels pandas Series.
                              
          outputs (pandas.Series):   Predictions' pandas array.
  
      Returns:
  
          Numpy array with MCC value if denominator is not equal to 0. Otherwise, 
          it returns None.
  
      """
      labels=labels.astype(np.bool_)
      outputs=outputs.astype(np.bool_)
      tp=np.logical_and(labels,outputs).astype(float).sum()
      tn=np.logical_and(np.logical_not(labels),np.logical_not(outputs)).astype(float).sum()
      fp=np.logical_and(np.logical_not(labels),outputs).astype(float).sum()
      fn=np.logical_and(labels,np.logical_not(outputs)).astype(float).sum()
      if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)>0:
          mcc=((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
      else:
          mcc=None
      return mcc

    def prc(self,labels,outputs):
      """
      It gets the Precision from a numpy array with labels and another with predictions.
   
      Args:
   
          labels  (pandas.Series):   Labels pandas Series.
                              
          outputs (pandas.Series):   Predictions' pandas array.
   
      Returns:
   
          Numpy array with the precision value if tp+fp are not equal to 0. Otherwise, 
          it returns None.
   
      """
      labels=labels.astype(np.bool_)
      outputs=outputs.astype(np.bool_)
      tp=np.logical_and(labels,outputs).astype(float).sum()
      fp=np.logical_and(np.logical_not(labels),outputs).astype(float).sum()
      if (tp+fp)>0:
          prc=tp/(tp+fp)
      else:
          prc=None
      return prc

    def rcc(self,labels,outputs):
      """
      It gets the Recall from a numpy array with labels and another with predictions.
   
      Args:
   
          labels  (pandas.Series):   Labels pandas Series.
                              
          outputs (pandas.Series):   Predictions' pandas array.
   
      Returns:
   
          Numpy array with the recall value if tp+fn are not equal to 0. Otherwise, 
          it returns None.
   
      """
      labels=labels.astype(np.bool_)
      outputs=outputs.astype(np.bool_)
      tp=np.logical_and(labels,outputs).astype(float).sum()
      fn=np.logical_and(labels,np.logical_not(outputs)).astype(float).sum()
      if (tp+fn)>0:
          rcc=tp/(tp+fn)
      else:
          rcc=None
      return rcc
