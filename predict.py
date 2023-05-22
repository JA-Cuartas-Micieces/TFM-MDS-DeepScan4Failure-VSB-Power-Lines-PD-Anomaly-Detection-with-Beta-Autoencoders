"""
MIT License
Copyright (c) 2021 HP-SCDS / Observatorio / Máster Data-Science UC /
Diego García Saiz / Jesús González Álvarez / Javier Alejandro Cuartas 
Micieces / 2021-2022 / DeepScan4Failure
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
sys.path.append(os.getcwd()+"/src")
import ds4f

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict on new data')
        
    parser.add_argument('--path',type=str, default=os.getcwd()+"/data", help='Path to the source data files')
    parser.add_argument('--data_file', help='Data file')
    parser.add_argument('--pr_loss', help='It returns loss function value apart from the expected output if True')
    parser.add_argument('--ths', help='Threshold to turn predictions from probabilities into binary')
    parser.add_argument('--load', help='Model name to load')
    parser.add_argument('--config', help='Scanner json configuration file name to load')
    parser.add_argument('--output_file', help='Required CSV file name for the predict output')
       
    args = parser.parse_args()
    
    if args.load not in os.listdir(args.path+"/models")+["False",None]:
        raise ValueError("Source autoencoder model for anomaly detection, {} not found in /models folder.".format(args.load))
        
    model = {"input_size":1,"z_samplesz":1,"hidden_sizes":[1,1,1]}
    
    scanner = ds4f.DeepScanner(model,
                               args.path,
                               load=args.load,
                               cname=args.config[:-17])
    
    res=scanner.predict(file=args.data_file,pr_loss=args.pr_loss,ths=args.ths) if args.ths!=None else scanner.predict_proba(file=args.path+args.data_file,pr_loss=args.pr_loss)
    
    pd.Series(res).to_csv(args.output_file)
        