"""MIT License
Copyright (c) 2023 HP-SCDS Observatorio 2021-2022 / Máster Data-Science UC /
Diego García Saiz / Javier Alejandro Cuartas Micieces / 2021-2022 / 
DeepScan4Failure

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
        