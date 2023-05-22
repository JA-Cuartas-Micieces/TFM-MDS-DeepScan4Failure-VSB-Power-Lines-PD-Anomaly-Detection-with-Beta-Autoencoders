import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
sys.path.append(os.getcwd()+"/src")
import ds4f
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--path',type=str, default=os.getcwd()+"/data", help='Path to the source data files')
    parser.add_argument('--data_file', help='Data file')
    parser.add_argument('--training_ids_file', help='Training ids file name')
    parser.add_argument('--validation_ids_file', help='Validation ids file name')
    parser.add_argument('--validation_lab_file', help='Validation labels file name')
    
    parser.add_argument('--load', help='Model name to load')
    parser.add_argument('--save_new', type=bool, default=False, help='Trained model name')
    parser.add_argument('--output_file', type=str, default="trainedScanner", help='Trained model name')
    parser.add_argument('--scanner_params_file',type=str,default=None,help='File name of the json file containing the desired scanner parameters in the /models directory, the inner dictionary key must.')
       
    args = parser.parse_args()
    
    if args.load not in os.listdir(args.path+"/models")+["False",None]:
        raise ValueError("Source autoencoder model for anomaly detection, {} not found in /models folder.".format(args.load))
        
    if args.scanner_params_file not in os.listdir(args.path+"/models")+["False",None]:
        raise ValueError("Source parameters for the scanner, {} not found in /models folder.".format(args.scanner_params_file))
        
    with open(args.path+"models/"+args.scanner_params_file, 'r') as f:
        g=json.load(f)
        if all([len(list(g[0].keys()))==1, args.scanner_params_file[:-5] in list(g[0].keys())]):
            model_file=list(g[0].keys())[0]
        else:
            raise ValueError("The key in the json file containing the parameters for the scanner must be the only one and the key must match the json file name.".format(args.scanner_params_file))
            
        nfeatures=g[0][model_file]["nfeatures"]
        batch_size=g[0][model_file]["batch_size"]
        epochs=g[0][model_file]["epochs"]
        beta=g[0][model_file]["beta"]
        thres_l=g[0][model_file]["thres_l"]
        lim_val_size=g[0][model_file]["lim_val_size"]
        limlossreadsepochtb=g[0][model_file]["limlossreadsepochtb"]
        directorytb=g[0][model_file]["directorytb"]
        early_stopping=g[0][model_file]["early_stopping"]
        z_samplesz=g[0][model_file]["m_lat_size"]
        input_size=g[0][model_file]["m_inp_size"]
        hidden_sizes=[int(l.strip()) for l in g[0][model_file]["m_hid_sizel"][1:-1].split(",")]
        optimizern=g[0][model_file]["optimizer"]
        optimizers_parameter_options=g[0][model_file]["optimizer_opts"]
        
    model = {"input_size":input_size,"z_samplesz":z_samplesz,"hidden_sizes":hidden_sizes}
    scanner = ds4f.DeepScanner(model,
                               args.path,
                               load=args.load if all([args.load!=None, args.load!="False"]) else False,
                               savenew=args.save_new,
                               cname=args.output_file)

    idtrain=pd.read_csv(args.path+"data/"+args.training_ids_file,index_col='Unnamed: 0').iloc[:,0]
    idvalidation=pd.read_csv(args.path+"data/"+args.validation_ids_file,index_col='Unnamed: 0').iloc[:,0]
    lbvalidation=pd.read_csv(args.path+"data/"+args.validation_lab_file,index_col='Unnamed: 0').iloc[:,0]
    
    scanner.fit(file=args.data_file,
                idtrain=idtrain,
                nfeatures=nfeatures,
                batch_size=batch_size,
                epochs=epochs,
                beta=beta,
                early_stopping=early_stopping,
                optimizer=optimizern,
                optimizer_params=optimizers_parameter_options,
                idvalidation=idvalidation,
                lbvalidation=lbvalidation,
                limlossreadsepochtb=limlossreadsepochtb,
                directorytb=directorytb)
    