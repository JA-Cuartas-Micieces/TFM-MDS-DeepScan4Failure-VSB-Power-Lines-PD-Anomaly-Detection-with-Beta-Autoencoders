# DeepScan4Failure

This is the implementation of an anomaly detection algorithm for big datasets, using Pytorch and Pyspark. It is part of the Data Science Master Thesis of Javier Cuartas, and It was guided by Walter Vinci, Ricardo Robles and Jesús González from HP-SCDS León 2021-2022, and by Diego García from the University of Cantabria.

Code is structured according to the following tree:
    
* **train.py** : python file that parse inputs from a shell prompt.
* **predict.py** : python file that parse inputs from a shell prompt.


* **models** : directory that contains model files (.pt format) and scanner configuration files (.json).
* **logs** : directory where all training logs will be stored. 
* **data** : directory where all input data will be stored.
* **runs** : directory where all tensorboard ouputs will be stored.
* **notebooks** : directory where validation jupyter notebooks will be stored.
    * **DS4F-Hyperparameter_tunning_example.ipynb**
    * **DS4F-Pretreatment-VSB-PowerLineFaultDetection.ipynb**


* **src** : module directory.
    * **\_\_init\_\_.py** : python file that makes this directory a module.
    * **dsf4.py** : python file to build scanners.
    * **model.py** : python file with the autoencoder models and all related items such as loss function...
    * **utils.py** : file with custom performance metrics for the scanner and the dataset modules built with pytorch library.
    
The python module has its own **dsf4.DeepScanner.fit()** and **dsf4.DeepScanner.predict()** methods as you can check printing the corresponding python native functions such as **dir()** or **help()**, but you can also find here an example on how to use the tool from a bash shell with the **train.py** and **predict.py** files.

**Fit**

This line takes data from **--data-file** (.h5), splits it into training and validation sets according to **--training_ids_file**, **--validation_ids_file** and **--validation_lab_file** (.csv) from **/data** directory, loads the existing model **--loads** (.pt) from the **/models** directory if It is specified, and trains a new model creating new json files if **--save_new** is set to True, using the **--output-file** name to create any new logs or derived files, and using as well the **--scanner_params_file** json configuration file in the **/models** directory, 

>!python train.py --help
>!python train.py --path "/home/ubuntu1/cdir/proj/" --data_file "scaled_3.h5" --training_ids_file "idtrain_2.csv" --validation_ids_file "idvalid_2.csv" --validation_lab_file "lbvalid_2.csv" --save_new True --output_file "intento_1" --scanner_params_file "trainedFromShell.json"

**Predict**

**--path** is used for the same purpose in predict.py as it was used in train.py. **--data_file** (.h5) is the file with the data we want to classify into normal/anomaly that must be placed into the **/data** directory, **--pr_loss** saves loss function values in addition to the normal/anomaly classification, **--ths** pass a probability threshold to the predict function so that it can output a binary outcome, **--load** refers to the model which will be used to perform the anomaly detection from the **/models** folder and **--output_file** is the file name of the csv with the input data classified into normal/anomaly.

>!python predict.py --help
>!python predict.py --path "/home/ubuntu1/cdir/proj/" --data_file "scaled_3.h5" --pr_loss True --ths 0.31 --load "model_intento1_20230521_165922.pt" --output_file "results_predict.csv" 