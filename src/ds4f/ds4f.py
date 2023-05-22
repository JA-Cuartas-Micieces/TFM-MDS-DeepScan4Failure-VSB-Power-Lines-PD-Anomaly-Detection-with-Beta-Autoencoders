"""
MIT License
Copyright (c) 2021 HP-SCDS / Observatorio / Máster Data-Science UC /
Diego García Saiz / Jesús González Álvarez / Javier Alejandro Cuartas 
Micieces / 2021-2022 / DeepScan4Failure

This module provides the tool main class (DeepScanner), with the common machine learning tool methods for training, performing inference on new data and loading models which were trained before. It is meant for semi-supervised learning in heavily imbalanced datasets, as it is based on Variational Autoencoders.

Functions:

    fit():            This method trains the chosen model with 
                      the chosen parameters. It saves a log in 
                      the logs folder, a .json parameters file 
                      in the models folder, a .pt model weights 
                      file also in the models folder, and a 
                      tensorboard folder in the runs directory. 
                      All input data must be located in the data 
                      folder. MCC (Mathew's Correlation Coefficient 
                      was selected as optimization metric since it 
                      takes into account discriminative power for 
                      imbalanced datasets.

    predict():        Performs classification on new data taking 
                      into account the DeepScanner instance and a 
                      probability threshold. All input data must be 
                      located in the data folder.

    predict_proba():  Performs inference on new data taking into 
                      account the DeepScanner instance model (non 
                      calibrated probability output). All input data
                      must be located in the data folder.

    load_model():     This method provides a way to load a previously
                      trained model along with its DeepScanner parameters
                      from the .pt and .json files in the ´models´ folder.

Classes:

    DeepScanner:      Anomaly detection tool based in a (Beta) Variational 
                      Autoencoder, ready to be tunned through several
                      available parameters.
"""

import os
import datetime
import time
import json
import math
import matplotlib as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .utils import *
from .model import *
from sklearn import metrics

class DeepScanner:
  """
  Main tool class. Anomaly detection tool based in a (Beta) Variational Autoencoder, ready to be tunned through several available parameters. It needs the rest of the package code to work which is available in 3 python modules, utils.py, model.py and the current ds4f.py, and It works with data in 4 directories, data, models, runs and logs.
  
  Attributes:
  
    path                    (str): Root path where all data folders are 
                                   located (data, models, runs and logs).

    logspath                (str): Path from root to logs folder.

    modelspath              (str): Path from root to models folder.
    
    directorytb             (str): Full path to tensorboard runs folders.
  
    device         (torch.device): Object refering to the device in which 
                                   torch Neural Network will be run.
  
    input_size              (int): Input size of the autoencoder network.
                  
    hidden_sizes           (list): List of sizes of the network, hidden 
                                   sizes from the input size (not included) 
                                   to the latent space size (included) of 
                                   the model.
                
    z_samplesz              (int): Number of samples taken from the latent 
                                   space each ´forward´ call of the 
                                   Autoencoder class instances (model 
                                   attribute in DeepScanner).
               
    model                (object): Autoencoder class instance.
  
    cname                   (str): Input name for the DeepScanner instance 
                                   and the derived data outputs such as log 
                                   files, model files, and tensorboard 
                                   folders.

    eval_fns               (dict): Dictionary with instances of loss error 
                                   evaluation functions from torch.nn, for 
                                   reporting purposes during training (only 
                                   ´CrossEntropyLoss´ available so far)
    
    eval_fn_n               (str): Selected key for eval_fns dictionary in the 
                                   current DeepScanner instance (only 
                                   ´CrossEntropyLoss´ available so far).

    nfeatures               (int): Number of columns per observation in the 
                                   training/inference dataset. Each observation 
                                   is defined by a set of different features 
                                   which were measured with the same frequency. 
                                   This number of consecutive dataset ids will 
                                   be concatenated and flattened as the 
                                   autoencoder input.
                           
    batch_size              (int): Established batch size for the training 
                                   torch DataLoader, so that this number of 
                                   ´nfeatures´ blocks are used in the training 
                                   loop.
                  
    epochs                  (int): Maximum number of epoch iterations for the
                                   training loop of the scanner.
                
    thres_l                (list): List of float numbers which will be checked as
                                   possible thresholds in the learning curve 
                                   validation loop during training for the fit() 
                                   method to provide with the best model 
                                   according to MCC metric (Mathew's Correlation 
                                   Coefficient).
                                   .
    beta                  (float): It is a regularization parameter in the Kullback
                                   Leiber part of the model loss function.
  
    early_stopping          (int): Regularization measure implementation. It is the 
                                   number of epochs the training loop will run 
                                   before breaking, from the moment the best 
                                   validation MCC achieved starts to be higher 
                                   than the current epoch validation MCCs
                
    lim_val_size          (float): Maximum proportion of the validation set batch 
                                   blocks to be used for learning curve building 
                                   purposes.
                                   
    limlossreadsepochtb     (int): Total number of learning validation reporting 
                                   loops run per epoch.
    
    savenew                (bool): If true, a new .json file will be created in
                                   the models folder to save the parameters of the 
                                   DeepScanner instance.
                                   
    optimizers             (dict): Dictionary with instances of optimizer functions 
                                   from torch.nn, for training purposes (only ´Adam´ 
                                   and ´SGD´ available so far)
    optimizers_parameter_options 
                           (dict): Dictionary with parameters for optimizer functions 
                                   from torch.nn, for training purposes (only ´Adam´ 
                                   and ´SGD´ available so far)
                  
    optimizern              (str): Selected key for ´optimizers´ and 
                                   ´optimizers_parameter_options´ dictionaries in the 
                                   current DeepScanner instance (only ´Adam´ and ´SGD´ 
                                   available so far).
                
    ths                     (int): Selected threshold for discrimination purposes 
                                   whenever ´predict´ method is called.

    file                    (str): Path to HDF-5 format input file. Group ´general/´
                                   expected. Dataset names must be ordered numbers. 
                                   Succesive column indexes must be grouped in columns 
                                   so that this Dataset class constructor can split 
                                   them into separate observations of nfeature columns. 
                                   This attribute refers only to the training loop input 
                                   data file with both training and validation samples.
    
    decisiontaker        (object): DecisionTaker class instance to perform inference 
                                   from both ´fit´ and ´predict´ methods.
    
    nrows                   (int): Total number of variables per signal (column) of 
                                   the .h5 file (or observations in each signal in the 
                                   original approach).
                
    ncolumns                (int): Total number of signals/time series in the dataset.

    g                      (dict): DeepScanner instance parameters dumped into a .json 
                                   file or read from a .json file.
    
  Methods:
      
    fit():            This method trains the chosen model with the chosen 
                      parameters. It saves a log in the logs folder,
                      a .json parameters file in the models folder, a .pt 
                      model weights file also in the models folder,
                      and a tensorboard folder in the runs directory. All 
                      input data must be located in the data folder.
                           
    predict():        Performs classification on new data taking into account 
                      the DeepScanner instance and a probability threshold. All 
                      input data must be located in the data folder.
                           
    predict_proba():  Performs inference on new data taking into account the 
                      DeepScanner instance model (non calibrated probability output). 
                      All input data must be located in the data folder.
                           
    load_model():     This method provides a way to load a previously trained
                      model along with its DeepScanner parameters
                      from the .pt and .json files in the ´models´ folder.
"""  
 
  def __init__(self,model,path,load=False,savenew=False,cname="trainedScanner"):
      """
      DeepScanner class constructor. It holds the common machine learning tool methods for training, for performing inference on new data and for loading models which were trained before. It is meant for semi-supervised learning in heavily imbalanced datasets, as it is based on Variational Autoencoders.

      Args:
      
        path      (str): Root path where all data folders are located (data, 
                         models, runs and logs).
                  
        model  (object): Autoencoder class instance.
        
      Kwargs:
      
        load (bool/str): It can contain False, if the user wants to train a
                         new model from scratch, or a file name ended with .pt
                         if the user wants to load an existing model to perform 
                         inference with it.
        
        savenew  (bool): If true, a new .json file will be created in the models 
                         folder to save the parameters of the DeepScanner 
                         instance.
                      
        cname     (str): Input name for the DeepScanner instance and the derived 
                         data outputs such as log files, model files, and 
                         tensorboard folders.
      """
      self.path=path
      self.logspath=path+"logs/"
      self.modelspath=path+"models/"
      self.directorytb=path+"runs/"
      
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.input_size=model["input_size"]
      self.hidden_sizes=model["hidden_sizes"]
      self.z_samplesz=model["z_samplesz"]
      self.model=Autoencoder(input_size=self.input_size,hidden_sizes=self.hidden_sizes,z_samplesz=self.z_samplesz).to(self.device)
      
      self.cname=cname
      self.eval_fns={"CrossEntropyLoss":torch.nn.CrossEntropyLoss()}
      self.eval_fn_n="CrossEntropyLoss"
      self.nfeatures=0
      self.batch_size=8
      self.epochs=20  
      self.thres_l=[0.15,0.3,0.45,0.6,0.75,0.9]
      self.beta=1
      self.early_stopping=10
      self.lim_val_size=1
        
      self.limlossreadsepochtb=2
      
      self.savenew=savenew
        
      self.optimizers_parameter_options={"SGD":{"lr":0.0001, "momentum":0, "dampening":0, "weight_decay":0, "nesterov":False, "maximize":False, "foreach":None},
                                  "Adam":{"lr":1e-3, "betas":(0.9, 0.999), "weight_decay":0, "eps":1e-8,"amsgrad":False, "maximize":False, "foreach":None,"capturable":False}}
      self.optimizern="Adam"
      self.ths=0.5
        
      if load!=False:
        self.load_model(load)

  def fit(self,file,idtrain,nfeatures,
          batch_size=None,epochs=None,
          beta=None,early_stopping=None,
          eval_fn_n=None,
          optimizer=None,optimizer_params=None,
          idvalidation=None,lbvalidation=None,
          thres_l=None,
          lim_val_size=None,
          limlossreadsepochtb=None,
          directorytb=None):      
      """
      Method for training the Variational Autoencoders model of the DeepScanner instance.

      Args:
      
        file               (str): Path to HDF-5 format input file. Group 
                                  ´general/´ expected. Dataset names must 
                                  be ordered numbers. Succesive column 
                                  indexes must be grouped in columns 
                                  so that this Dataset class constructor 
                                  can split them into separate observations 
                                  of nfeature columns. This attribute refers 
                                  only to the training loop input data 
                                  file with both training and validation 
                                  samples.
                                   
        idtrain  (pandas.Series): Pandas Series containing training set 
                                  indexes from the HDF-5 file content.
        
        nfeatures          (int): Number of columns per observation in the 
                                  training/inference dataset. Each observation 
                                  is defined by a set of different features 
                                  which were measured with the same frequency. 
                                  This number of consecutive dataset ids will 
                                  be concatenated and flattened as the 
                                  autoencoder input.
      Kwargs:
      
        batch_size         (int): Established batch size for the training torch 
                                  DataLoader, so that this number of ´nfeatures´ 
                                  blocks are used in the training loop.
                      
        epochs             (int): Maximum number of epoch iterations for the 
                                  training loop of the scanner.
                                          .
        beta             (float): It is a regularization parameter in the Kullback 
                                  Leiber part of the model loss function.
      
        early_stopping     (int): Regularization measure implementation. It is 
                                  the number of epochs the training loop will 
                                  run before breaking, from the moment the best 
                                  validation MCC achieved starts to be higher 
                                  than the current epoch validation MCCs.
                      
        eval_fn_n          (str): Selected key for eval_fns dictionary in the 
                                  current DeepScanner instance (only 
                                  ´CrossEntropyLoss´ available so far).
                                      
        optimizer          (str): Selected key for ´optimizers´ and 
                                 ´optimizers_parameter_options´ dictionaries 
                                 in the current DeepScanner instance (only ´Adam´ 
                                 and ´SGD´ available so far).

        optimizer_paramns (dict): Dictionary with parameters for optimizer 
                                  functions from torch.nn, for training purposes 
                                  (only ´Adam´ and ´SGD´ available so far)
                                       
        idvalidation (pd.Series): Pandas Series containing validation set indexes 
                                  of the HDF-5 file content.
        
        lbvalidation (pd.Series): Pandas Series containing validation ground truth 
                                  corresponding to the validation set. 
        
        thres_l           (list): List of float numbers which will be checked as 
                                  possible thresholds in the learning curve validation 
                                  loop during training for the fit() method to 
                                  provide with the best model according to MCC metric 
                                  (Mathew's Correlation Coefficient).
                                                       
        lim_val_size     (float): Maximum proportion of the validation set batch blocks 
                                  to be used for learning curve building purposes.
                                       
        limlossreadsepochtb(int): Total number of learning validation reporting loops 
                                  run per epoch.
        
        directorytb        (str): Full path to tensorboard runs folders.
                
      Raise:
      
        A ValueError exception will be raised whenever optimizer function parameters 
        passed are incomplete. 
                
        A ValueError exception will be raised whenever input data file format is not 
        h5.
        
      """

      self.file=self.path+"data/"+file
      optimizer=optimizer if optimizer!=None else self.optimizern
      optimizer_params=optimizer_params if optimizer_params!=None else self.optimizers_parameter_options[optimizer]

      for el in self.optimizers_parameter_options.keys():
        for pel in self.optimizers_parameter_options[el].keys():
          if pel not in optimizer_params.keys():
            if (el==optimizer):
              raise ValueError("Optimizer function parameters incomplete, check pytorch documentation and use all possible parameters in the dictionary: https://pytorch.org/docs/stable/optim.html.")
            else:
              optimizer_params[pel]=self.optimizers_parameter_options[el][pel]

      self.optimizers={"SGD":torch.optim.SGD(self.model.parameters(),lr=optimizer_params["lr"],momentum=optimizer_params["momentum"],dampening=optimizer_params["dampening"], weight_decay=optimizer_params["weight_decay"], 
                                      nesterov=optimizer_params["nesterov"], maximize=optimizer_params["maximize"], foreach=optimizer_params["foreach"]),
                "Adam":torch.optim.Adam(self.model.parameters(),optimizer_params["lr"], betas=optimizer_params["betas"], eps=optimizer_params["eps"], weight_decay=optimizer_params["weight_decay"], amsgrad=optimizer_params["amsgrad"], 
                                        foreach=optimizer_params["foreach"], maximize=optimizer_params["maximize"], capturable=optimizer_params["capturable"])}
        
      self.optimizer=self.optimizers[optimizer]
            
      eval_fn=self.eval_fns[eval_fn_n] if eval_fn_n!=None else self.eval_fns[self.eval_fn_n]
      batch_size=self.batch_size if batch_size==None else batch_size
      epochs=self.epochs if epochs==None else epochs
      beta=self.beta if beta==None else beta
      early_stopping=self.early_stopping if early_stopping==None else early_stopping
      thres_l=self.thres_l if thres_l==None else thres_l
      lim_val_size=self.lim_val_size if lim_val_size==None else lim_val_size
      directorytb=self.directorytb if directorytb==None else directorytb
        
      limlossreadsepochtb=self.limlossreadsepochtb if limlossreadsepochtb==None else limlossreadsepochtb

      timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
      logsfn="{}{}_log.log".format(self.logspath,timestamp)
      logft = open(logsfn, "w")
      logft.close()
    
      self.ncolumns=len(idvalidation)+len(idtrain) if isinstance(lbvalidation,pd.Series) else len(idtrain)
      idn = str(np.random.default_rng(53168).integers(low=0, high=len(idtrain), size=1)[0])
      if "h5" in self.file.split("."):
        with h5py.File(self.file, "r") as f:
          nf=f["general/"+str(idn)][:]
          nr=pd.DataFrame(nf[:,:].tolist()[0])
      else:
        raise ValueError("Input data file extension not recognised, convert the training set data to a .h5 file, please.")
      self.nrows=nr.shape[0]
      self.nfeatures=nfeatures
      del nr

      self.limlossreadsepochtb=limlossreadsepochtb
      self.directorytb=directorytb
      self.thres_l=thres_l
      self.early_stopping= early_stopping
      self.beta=beta
      self.epochs=epochs
      
      if isinstance(idvalidation,pd.Series):
        self.lim_val_size=lim_val_size
        lim_v=int(math.floor(lim_val_size*(len(idvalidation))/(self.nfeatures*batch_size)))
        while (lim_v)%(self.nfeatures*batch_size)!=0:
          lim_v=lim_v-1
        lim_v=1 if lim_v==0 else lim_v

        self.thres_l=thres_l

        #######################################################
        #VALIDATION SET PRETREATMENT FOR THE ALGORITHM TO WORK#
        #######################################################

        tmadjustingval0=time.time()
        # Adjusting validation set in order to get first, one label of anomaly/non-anomaly 
        # per observation, since the original dataset might have anomaly/non-anomaly labels 
        # per feature instead of per observation (group of features).
        trunif0=time.time()
        lbvalidation=pd.concat([lbvalidation, idvalidation,pd.Series(np.repeat(np.arange(len(idvalidation)/self.nfeatures),self.nfeatures).astype(np.int32)).rename("labref")], axis=1).reset_index()
        nd=pd.Series(np.array((lbvalidation.loc[:,["target","labref"]].groupby("labref").sum()>0)["target"]).astype(np.int32)).rename("targetr")
        lbvalidation=pd.concat([lbvalidation,np.repeat(nd,self.nfeatures).reset_index()["targetr"]],axis=1)
        trunif=time.time()-trunif0

        # Adjusting validation set in order to get batches with the same or pretty close 
        # number of anomalies vs non-anomalies for the calculation of the training error 
        # part of the learning curve. 
        rsplit0=time.time()
        a=lbvalidation.loc[lbvalidation["targetr"]==1,:].reset_index()
        b=lbvalidation.loc[lbvalidation["targetr"]==0,:].reset_index()
        d = list()
        i0=0
        i1=0
        rsplit=time.time()-rsplit0
        r0loop=time.time()
        la=int((len(a)/self.nfeatures)/(len(lbvalidation)/(batch_size*self.nfeatures)))
        lb=batch_size-la
        d=[]
        for i in np.arange(len(lbvalidation)/(batch_size*self.nfeatures)):
          if i!=(len(lbvalidation)/(batch_size*self.nfeatures)-1):
            d=d+a.loc[i0*self.nfeatures:(i0*self.nfeatures+la*self.nfeatures-1),"signal_id"].to_list()+b.loc[i1*self.nfeatures:(i1*self.nfeatures+lb*self.nfeatures-1),"signal_id"].to_list()
          else:
            d=d+a.loc[i0*self.nfeatures:(len(a)-1),"signal_id"].to_list()+b.loc[i1*self.nfeatures:(len(b)-1),"signal_id"].to_list()
          i0=i0+la
          i1=i1+lb
        tloop=time.time()-r0loop
        tresort0=time.time()
        lbvalidation = lbvalidation.set_index('signal_id')
        lbvalidation=lbvalidation.loc[d]
        tresort=time.time()-tresort0
        idvalidation=lbvalidation.index.to_series()
        idvalidation=idvalidation.reset_index(drop=True)
        lbvalidation=lbvalidation.loc[:,"targetr"]
        lbvalidation=lbvalidation.iloc[0:(len(idvalidation)+1):self.nfeatures]
        del i0,i1,a,b,nd,d, lb, la
        tradj=time.time()-tmadjustingval0

      ################################################
      #COUNTERS, PROPERTIES AND LOCAL VARIABLES SETUP#
      ################################################

      writer = SummaryWriter('{}{}_{}'.format(directorytb,self.cname,timestamp))
      self.decisiontaker=DecisionTaker()

      best_mccv = -1

      # Position counters for training curve.
      itb=0
      ivb=0
      irb=0

      # Position counters for learning curves.
      ivts=0
      ivs=0

      # Time counters for learning curves.
      cttot=0
      ctr_vv=0
      ctr_vt=0
      ctr_tl=0

      # Loss counters for learning curves and decision over outlier/non-outlier nature.
      avgt_loss=0
      avgv_loss=0

      # Early stopping counter.
      e_stop_c=0
      best_ref_ep=-1
    
      ##############
      #DATA LOADING#
      ##############

      if "h5" in self.file.split("."):
        trtot0=time.time()
        dataset_train = H5LDataset(
            ids=idtrain,
            file=self.file,
            nrows=self.nrows,
            ncolumns=self.ncolumns,
            nfeatures=self.nfeatures,
            transform=torch.tensor)
        training_loader=DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        t0t0=time.time()-trtot0
        if isinstance(idvalidation,pd.Series):
          dataset_val = H5LDataset(
            ids=idvalidation,
            file=self.file,
            nrows=self.nrows,
            ncolumns=self.ncolumns,
            nfeatures=self.nfeatures,
            transform=torch.tensor)
          val_loader=DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
          t0v0=time.time()-trtot0-t0t0

      else:
        raise ValueError("Input data file extension not recognised, convert the training set data to a .h5 file, please.")

      logft = open(logsfn, "a")

      ######################
      #SCANNER RECORD SETUP#
      ######################
      model_name = 'model_{}_{}.pt'.format(self.cname, timestamp)
      if not os.path.exists('{}savedmodels.json'.format(self.modelspath+self.cname+"_")):
          with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'w') as cr_f:
            json.dump([{"{}".format(model_name):{}}],cr_f)
          with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'r') as f:
            self.g=json.load(f)
      else:
          if self.savenew==True:
            for inc in range(0,1000): 
              if self.cname+"_"+str(inc)+"_savedmodels.json" not in os.listdir():
                self.cname=self.cname+"_"+str(inc)
                break
            with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'w') as cr_f:
              json.dump([{"{}".format(model_name):{}}],cr_f)
            with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'r') as f:
              self.g=json.load(f)
          else:
            with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'r') as f:
              self.g=json.load(f)
    
      #######################
      #INITIAL TIMES SUMMARY#
      #######################

      logft.write("\n--------------------------")
      logft.write("\nARCHITECTURE")
      logft.write("\n--------------------------")
      logft.write("\n")
      logft.write("********************************")
      logft.write("\n")
      logft.write(str(self.model.layerlst))
      logft.write("\n")
      logft.write("********************************")
      logft.write("\n")
      logft.write("\n--------------------------")
      logft.write("\nSET UP PARAMETERS")
      logft.write("\n--------------------------")
      logft.write("\nnfeatures: {}".format(str(self.nfeatures)))
      logft.write("\nepochs: {}".format(str(self.epochs)))
      logft.write("\nbeta: {}".format(str(self.beta)))
      logft.write("\nthres_l: {}".format(str(self.thres_l)))
      logft.write("\nlim_val_size: {}".format(str(self.lim_val_size)))
      logft.write("\nlimlossreadsepochtb: {}".format(str(self.limlossreadsepochtb)))
      logft.write("\ndirectorytb: {}".format(str(self.directorytb)))
      logft.write("\nearly_stopping: {}".format(str(self.early_stopping)))
      logft.write("\n--------------------------")
      logft.write("\nSET UP TIMES")
      logft.write("\n--------------------------")
      if isinstance(idvalidation,pd.Series):
        logft.write('\nTime Validation Batch Resampling to have both anomaly and non-anomaly classes: {} min.'.format(str(round(tradj/60,2))))
        logft.write('\nTime Validation set loading: {} min.'.format(str(round(t0v0/60,2))))
      logft.write('\nTime Training set loading: {} min.'.format(str(round(t0t0/60,2))))
      logft.write("\n--------------------------")

      ############
      ############
      #EPOCH LOOP###############################################################
      ############
      ############

      for epoch in range(epochs):
          ###############################################
      #####EPOCH LOOP COUNTERS AND LOCAL VARIABLES SETUP#
          ###############################################

          # Time variable initialization for Global per epoch reporting 
          ttot0=time.time()
          # Time variable initialization for Training Loop execution (per epoch reporting) 
          tr_tl=0 
          # Time variable initialization for Learning Curve building on the Training Set (per epoch reporting)
          tr_vt=0
          # Time variable initialization for Learning Curve building and performance follow up on the Validation Set (per epoch reporting)
          tr_vv=0

          logft.write("\n=========")
          logft.write('\nEPOCH {}:'.format(epoch + 1))
          logft.write("\n=========\n")

          # Setting epoch reporting readings to the limit we pass as an argument
          # for the reporting of training loss

          rj=math.floor(len(idtrain)/(self.nfeatures*batch_size*limlossreadsepochtb)) if math.floor(len(idtrain)/(self.nfeatures*batch_size*limlossreadsepochtb))>1 else 1

          training_loss = 0
          validation_loss = 0
          trloss=0

          ######################
      #####EPOCH TRAINING LOOP #
          ######################
          
          logft.write("\n   TRAINING") 
          for i,data in enumerate(training_loader):
            t0=time.time()
            logft.write("\n     TRAINING STEP nº{}".format(i+1))
            inputs,ids = data
            inputs=inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.model.loss_fn(outputs, inputs,beta,"t")
            loss.backward()
            self.optimizer.step()

            # Data gathering for reporting training loss
            trloss+=loss.item()
            if (i!=0) and ((i+1) % rj == 0) and irb<limlossreadsepochtb*(epoch+1):
              last_rloss = trloss / rj if i>0 else trloss
              writer.add_scalar('Training Curve:', round(last_rloss,2), irb)
              writer.flush()
              trloss=0
              with torch.no_grad():
                if isinstance(idvalidation,pd.Series):
                    logft.write("\n       TRAINING ANOMALY DETECTION REFERENCE")
                    tr_tl=tr_tl+time.time()-t0

              ###################################################
      #########EPOCH VALIDATION AND LEARNING CURVE BUILDING LOOP#
              ###################################################

                    # EPOCH Learning curve building loop from the training set data:
                    #--------------------------------------------------------------#
                    t0t=time.time()
                    logft.write("\n       BUILDING LEARNING CURVE - TRAINING SET")

                    for itv,datat in enumerate(training_loader):
                      if itv>=ivts:
                        inputst,idst = datat
                        inputst=inputst.to(self.device)
                        outputst = self.model(inputst)
                        losst = self.model.loss_fn(outputst, inputst,beta,"v")
                        rest=self.decisiontaker.run([outputst,inputst])
                      #  tloss_res=eval_fn(rest.float(),torch.tensor(np.repeat([0],len(list(rest))), dtype=torch.float32))
                        tloss_res=losst.mean().item()
                        training_loss+=tloss_res

                      # trainres variable will keep the output of the anomaly identifier 
                      # to build learning curves and report performance scores afterwards
                        if itv==ivts:
                          trainres=rest.clone()
                        else:
                          trainres=torch.cat([trainres,rest.clone()])
                          
                    # lim_val_size time limit for validation is taken into account in this 
                    # last part of the loop 
                      if itv==ivts+(lim_v-1):
                        ivts=ivts+lim_v
                        if ivts>=int(len(idvalidation)/(self.nfeatures*batch_size)):
                          ivts=0
                        break

                    tr_vt=tr_vt+time.time()-t0t

                    # EPOCH Validation Run and Learning curve building loop from the Validation set data:
                    #-----------------------------------------------------------------------------------#
                    t0v=time.time()
                    logft.write("\n       BUILDING LEARNING CURVE - VALIDATION SET")
                    for iv,datav in enumerate(val_loader):
                      if iv>=ivs:
                        inputsv,idv = datav
                        inputsv=inputsv.to(self.device)
                        outputsv = self.model(inputsv)
                        lossv = self.model.loss_fn(outputsv, inputsv,beta,"v")
                        resv=self.decisiontaker.run([outputsv, inputsv])
                      #  tloss_resv=eval_fn(resv.float(),torch.tensor(lbvalidation.iloc[(batch_size*iv):(batch_size*(iv+1))].to_numpy(), dtype=torch.float32)).numpy()
                        tloss_resv=lossv.mean().item()
                        validation_loss+=tloss_resv

                      # valres variable will keep the output of the anomaly identifier 
                      # to build learning curves and report performance scores afterwards

                        if iv==ivs:
                          valres=resv.clone()
                        else:
                          valres=torch.cat([valres,resv.clone()])

                    # lim_val_size time limit for validation is taken into account in this 
                    # last part of the loop through lim_vels, that was defined on the 
                    # previous part for building the learning curve with the training set.

                      if iv==ivs+(lim_v-1):
                        ivs=ivs+lim_v
                        if ivs>=int(len(idvalidation)/(self.nfeatures*batch_size)):
                          ivs=0
                        break

                    tr_vv=tr_vv+time.time()-t0v
                    avgt_loss=training_loss/lim_v
                    training_loss=0
                    avgv_loss=validation_loss/lim_v
                    validation_loss=0
                    mccv=-1

                    thsv=0.5
                    auv=metrics.roc_auc_score(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),valres.numpy()) if all([lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy().sum()!=0,lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy().sum()!=batch_size*lim_v]) else 0.5
                    mccv=self.decisiontaker.mcc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>0.5).astype(np.int32))
                    mccv=mccv if mccv!=None else -1
                    acv=metrics.accuracy_score(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>0.5).astype(np.int32)) if all([lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy().sum()!=0,lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy().sum()!=batch_size*lim_v]) else 0.5
                    rcv=self.decisiontaker.rcc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>0.5).astype(np.int32))
                    rcv=rcv if rcv!=None else 0
                    prv=self.decisiontaker.prc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>0.5).astype(np.int32))
                    prv=prv if prv!=None else 0
                    
                    for ths in self.thres_l:
                        
                      smcc=self.decisiontaker.mcc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>ths).astype(np.int32))
                      sacv=metrics.accuracy_score(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>ths).astype(np.int32))
                      srcv=self.decisiontaker.rcc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>ths).astype(np.int32))
                      sprv=self.decisiontaker.prc(lbvalidation.iloc[0:(batch_size*lim_v)].to_numpy(),(valres.numpy()>ths).astype(np.int32))

                      if smcc!=None and smcc>mccv:
                        thsv=ths
                        mccv=smcc
                        acv=sacv
                        rcv=srcv if srcv!=None else 0
                        prv=sprv if sprv!=None else 0
                    
                # Tensorboard Learning Curve Reporting for this epoch's performance metrics
                
                    writer.add_scalars('Training vs Validation Loss',
                                    { 'Training Loss' : avgt_loss, 'Validation Loss' : avgv_loss },
                                    irb)
                    writer.add_scalars('Validation MCC',
                                    {'Validation MCC' :mccv},
                                    irb)
                    writer.add_scalars('Validation AUC-ROC vs MCC',
                                    {'Validation MCC': mccv, 'Validation AUC-ROC' : auv},
                                    irb)
                    writer.add_scalars('Validation Accuracy, Precision and Recall',
                                    { 'Validation Accuracy' : acv,'Validation Precision' : prv,  'Validation Recall' : rcv},
                                    irb)

                    writer.flush()

                # Track best performance if this epoch scanner is better than the previous, 
                # and save the model's state.
                    if any([mccv >= best_mccv,epoch==0]):
                      best_ths=thsv
                      best_vloss = avgv_loss
                      best_tloss=avgt_loss
                      
                      if os.path.exists(self.modelspath+model_name):
                        os.remove(self.modelspath+model_name)
                      torch.save(self.model.state_dict(), self.modelspath+model_name)
                      
                      best_aucv=auv
                      best_mccv=mccv
                      best_acc=acv
                      best_pre=prv
                      best_rec=rcv
                      best_epc=epoch
                        
              irb+=1

            else:
              tr_tl=tr_tl+time.time()-t0
              
          ################################################
      #####EPOCH EXECUTION TIME AND PERFORMANCE REPORTING#
          ################################################

          ttot=time.time()-ttot0

          cttot+=ttot
          ctr_vv+=tr_vv
          ctr_vt+=tr_vt
          ctr_tl+=tr_tl

          logft.write("\n--------------------------")
          logft.write('\n   EPOCH TIMES:')
          logft.write("\n--------------------------")
          
          if isinstance(idvalidation,pd.Series):
            logft.write('\n     Cumulative Time Distribution: train {} %, trainset valid {} %, valset valid {} %.'.format(str(round(100*(ctr_tl+t0t0/2)/(cttot+t0t0+t0v0),2)),str(round(100*(ctr_vt+t0t0/2)/(cttot+t0t0+t0v0),2)), str(round(100*(ctr_vv+t0v0)/(cttot+t0t0+t0v0),2))))
          logft.write('\n     Epoch Time Training: {} min.'.format(str(round(tr_tl/60,2))))
          if isinstance(idvalidation,pd.Series):
            logft.write('\n     Epoch Time Validation: {} min.'.format(str(round((tr_vt+tr_vv)/60,2))))
            logft.write('\n     Epoch Time Total: {} min.'.format(str(round(ttot/60,2))))
            logft.write("\n----------------------------------")
            logft.write('\n   BEST PERFORMANCE SO FAR:')
            logft.write("\n----------------------------------")
            logft.write('\n     Epoch Learning Loss: train {} valid {}'.format(str(round(best_tloss,2)), str(round(best_vloss,2))))
            logft.write('\n     Epoch Learning threshold: {}'.format(str(round(best_ths,2))))
            logft.write('\n     Epoch Learning EPOCH Nº: {}'.format(str(round(best_epc,2))))
            logft.write('\n     Epoch Learning AUC: valid {}'.format(str(round(best_aucv,2))))
            logft.write('\n     Epoch Learning MCC: valid {}'.format(str(round(best_mccv,2))))
            logft.write('\n     Epoch Learning Accuracy: valid {}'.format(str(round(best_acc,2))))
            logft.write('\n     Epoch Learning Precision: valid {}'.format(str(round(best_pre,2))))
            logft.write('\n     Epoch Learning Recall: valid {}'.format(str(round(best_rec,2))))
        
          if isinstance(idvalidation,pd.Series):
            e_stop_c=e_stop_c+1 if best_mccv<=best_ref_ep else 0
            best_ref_ep=best_mccv if best_mccv>=best_ref_ep else best_ref_ep
            if self.early_stopping!=None and e_stop_c>=self.early_stopping:
              break
              


      ############
      ############
      #EPOCH LOOP###############################################################
      ############
      ############

      writer.close()

      #################################################################################################################
      #SAVING PERFORMANCE, TIME AND KEY VARIABLES FOR THE BEST TRAINED MODEL BASING REFERENCE ON THE FULL TRAINING SET#
      #################################################################################################################
      
      self.g[0][model_name]={}
      self.g[0][model_name]["batch_size"]=batch_size
      self.g[0][model_name]["epochs"]=epochs
      self.g[0][model_name]["limlossreadsepochtb"]=limlossreadsepochtb
      self.g[0][model_name]["directorytb"]=directorytb
      self.g[0][model_name]["nfeatures"]=nfeatures
      self.g[0][model_name]["beta"]=beta
      self.g[0][model_name]["m_inp_size"]=self.input_size
      self.g[0][model_name]["m_hid_sizel"]=str(self.hidden_sizes)
      self.g[0][model_name]["m_lat_size"]=self.z_samplesz
      self.g[0][model_name]["training_batches"]=len(training_loader)
      self.g[0][model_name]["time_initial_training_set_loading"]=round(t0t0/60,2)
      self.g[0][model_name]["optimizer"]=optimizer
      self.g[0][model_name]["optimizer_opts"]=optimizer_params
        
      if isinstance(idvalidation,pd.Series):
          self.g[0][model_name]["epoch"]=best_epc
          self.g[0][model_name]["thres_l"]=thres_l
          self.g[0][model_name]["early_stopping"]=early_stopping
          self.g[0][model_name]["validation_batches"]=len(val_loader)
          self.g[0][model_name]["time_epoch_avg(min)"]=round(cttot/(epochs*60),2)
          self.g[0][model_name]["time_validation_avg(min)"]=round((ctr_vt+ctr_vv)/(epochs*60),2)  
          self.g[0][model_name]["time_training_avg(min)"]=round(ctr_tl/(epochs*60),2)
          self.g[0][model_name]["time_initial_validation_pre_sorting"]=round(tradj/60,2)
          self.g[0][model_name]["time_initial_validation_set_loading"]=round(t0v0/60,2)
          self.g[0][model_name]["v_avg_loss"]=best_vloss
          self.g[0][model_name]["v_threshold"]=best_ths
          self.g[0][model_name]["v_auc"]=best_aucv
          self.g[0][model_name]["v_mcct"]=best_mccv
          self.g[0][model_name]["v_mccv"]=best_mccv
          self.g[0][model_name]["v_accuracy"]=best_acc
          self.g[0][model_name]["v_precision"]=best_pre
          self.g[0][model_name]["v_recall"]=best_rec
          self.g[0][model_name]["lim_val_size"]=self.lim_val_size
          self.g[0][model_name]["eval_fn"]=self.eval_fn_n

      with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'w') as cr_f:
          json.dump(self.g,cr_f)

      if isinstance(idvalidation,pd.Series):
          logft.write("\n=========")
          logft.write('\nEND')
          logft.write("\n=========\n")
          logft.write('\n\n   BEST MODEL PERFORMANCE:')
          logft.write('\n     Loss validation: {}'.format(str(best_vloss))) 
          logft.write('\n     Threshold: {}'.format(str(best_ths)))
          logft.write('\n     MCC validation: {}'.format(str(best_mccv)))
          logft.write('\n     AUC validation: {}'.format(str(best_aucv)))
          logft.write('\n     Accuracy validation: {}'.format(str(best_acc)))
          logft.write('\n     Precision validation: {}'.format(str(best_pre))) 
          logft.write('\n     Recall validation: {}'.format(str(best_rec)))
          
      logft.close()
    
  def load_model(self,model_file):      
      """
      Method to load a previously trained DeepScanner object, including an existing Variational Autoencoders model and the parameters for the DeepScanner instance which were stored in a .json file of the same folder (models). For the DeepScanner parameters, the content of the working DeepScanner instance attribute ´cname´ will be considered.

      Args:
      
        modelfile (object): model file name including its extension. 
                            The model weights file .pt must be located 
                            on models directory, as well as the DeepScanner 
                            .json file with the desired parameters with a 
                            name including the content of the working 
                            DeepScanner instance attribute ´cname´.
      """

      with open('{}savedmodels.json'.format(self.modelspath+self.cname+"_"), 'r') as f:
      
        self.g=json.load(f)
        self.nfeatures=self.g[0][model_file]["nfeatures"]
        self.batch_size=self.g[0][model_file]["batch_size"]
        self.epochs=self.g[0][model_file]["epochs"]
        self.beta=self.g[0][model_file]["beta"]
        self.thres_l=self.g[0][model_file]["thres_l"]
        self.lim_val_size=self.g[0][model_file]["lim_val_size"]
        self.limlossreadsepochtb=self.g[0][model_file]["limlossreadsepochtb"]
        self.directorytb=self.g[0][model_file]["directorytb"]
        self.early_stopping=self.g[0][model_file]["early_stopping"]
        self.eval_fn_n=self.g[0][model_file]["eval_fn"]
        self.z_samplesz=self.g[0][model_file]["m_lat_size"]
        self.input_size=self.g[0][model_file]["m_inp_size"]
        self.ths=self.g[0][model_file]["v_threshold"]
        self.hidden_sizes=[int(l.strip()) for l in self.g[0][model_file]["m_hid_sizel"][1:-1].split(",")]
        self.optimizern=self.g[0][model_file]["optimizer"]
        self.optimizers_parameter_options[self.optimizern]=self.g[0][model_file]["optimizer_opts"]
        
      self.model=Autoencoder(input_size=self.input_size,hidden_sizes=self.hidden_sizes,z_samplesz=self.z_samplesz)
      self.model.load_state_dict(torch.load(self.modelspath+model_file))
           
  def predict(self,file=None,dataset=None,pr_loss=True,ths=None):      
      """
      Method for performing classification between the cathegories outlier/normal from a DeepScanner instance.

      Kwargs:
      
        file      (str): HDF-5 format file with the data user wants to 
                         classify. It can also contain None if a torch 
                         dataset object is provided instead. (See file 
                         attribute above).
                  
        dataset(object): torch Dataset instance with the same requirements 
                         as H5LDataset class docstring states (See utils.py 
                         H5LDataset class). It can also contain None if a 
                         .h5 file is provided instead.
                         
        pr_loss  (bool): This flag sets up whether the user wants the method 
                         to return a loss value per sample, or just wants the 
                         prediction ouput.
        
        ths     (float): Theshold float to turn ´predict_proba´ output 
                         probabilities into binary predictions.
        
      Returns:
      
        Depending on pr_loss value, It will output two arrays with losses and 
        predictions, or just the last one.
        
      Raise:
      
        An exception will be raised if the input file is not .h5 format.
        
      """
  
      ths=self.ths if ths==None else ths
      rs = self.predict_proba(file=file,dataset=dataset,pr_loss=pr_loss)
      res = (rs[1]>ths) if pr_loss==True else rs
    
      if pr_loss==True:
        return rs[0],res
      else:
        return res
    
  def predict_proba(self,file=None,dataset=None,pr_loss=True):      
  
      """
      Method for performing probabilities inference, from a DeepScanner instance.

      Kwargs:
      
        file      (str): HDF-5 format file with the data user wants to classify. 
                         It can also contain None if a torch dataset object 
                         is provided instead. (See file attribute above).
                  
        dataset(object): torch Dataset instance with the same requirements as 
                         H5LDataset class docstring states (See utils.py 
                         H5LDataset class). It can also contain None if a .h5 
                         file is provided instead.
                         
        pr_loss  (bool): This flag sets up whether the user wants the method to 
                         return a loss value per sample, or just wants the 
                         prediction ouput.
        
      Returns:
      
        Depending on pr_loss value, It will output two arrays with losses and 
        predictions, or just the last one.
        
      Raise:
      
        An exception will be raised if the input file is not .h5 format.
        
      """
    
      if dataset==None:
      
        file=self.path+"data/"+file

        if "h5" in file.split("."):
            
          with h5py.File(file, "r") as f:
            dg=f["general"]
            idg=[int(t) for t in dg.keys()]
            
          ids=sorted(idg)
          idn = str(np.random.default_rng(12345).integers(low=0, high=len(ids), size=1)[0])

          with h5py.File(file, "r") as f:
            nf=f["general/"+str(idn)][:]
            nr=pd.DataFrame(nf[:,:].tolist()[0])
            
        else:
          raise ValueError("Input data file extension not recognised, convert the training set data to a .h5 file, please.")

        self.nrows=nr.shape[0]
        self.ncolumns=len(idn)

        dataset = H5LDataset(
          ids=ids,
          file=file,
          nrows=self.nrows,
          ncolumns=self.ncolumns,
          nfeatures=self.nfeatures,
          transform=torch.tensor)
        
      self.decisiontaker=DecisionTaker()
      loader=DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

      res=list()
      loss=list()
      with torch.no_grad():
        for i,data in enumerate(loader):
          inputst,idst = data
          inputst=inputst.to(self.device)
          outputst = self.model(inputst)
          losst = self.model.loss_fn(outputst, inputst,self.beta,"v")
          rest=self.decisiontaker.run([outputst,inputst])
          res=torch.cat([res,rest.clone()]) if i!=0 else rest
          lss=torch.cat([lss,rest.clone()]) if i!=0 else losst
    
      res=res.numpy()
      lss=lss.numpy()
    
      if pr_loss==True:
        return lss,res
      else:
        return res

