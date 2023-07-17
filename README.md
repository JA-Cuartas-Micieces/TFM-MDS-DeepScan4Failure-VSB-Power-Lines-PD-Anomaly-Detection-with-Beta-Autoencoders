### PROJECT

This is the implementation of an anomaly detection algorithm, using Pytorch. It is part of the Data Science Master Thesis of Javier Cuartas, and It was guided by HP-SCDS León 2021-2022, and by Diego García from the University of Cantabria.

Code is structured according to the following tree:
    
* **train.py** : python file that parse inputs from a shell prompt.
* **predict.py** : python file that parse inputs from a shell prompt.


* **models** : directory that contains model files (.pt format) and scanner configuration files (.json).
* **logs** : directory where all training logs will be stored. 
* **data** : directory where all input data will be stored.
* **runs** : directory where all tensorboard ouputs will be stored.
* **docs** : directory where the written part of this work is stored.
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

### CONTACT AND CONTRIBUTION

It is a personal project which is not open for contributions at the moment but please, feel free to share any comments, questions or suggestions through javiercuartasmicieces@hotmail.com.

### ACKNOWLEDGEMENTS

* Addison H, Dane S, Vantuch T. Power Line Fault Detection (2018). Kaggle [Online]. [Available](https://www.kaggle.com/competitions/vsb-power-line-fault-detection/data.)

* An J, Cho S. “Variational Autoencoder based Anomaly Detection using Reconstruction Probability”. Special Lecture on IE. 2015. Vol. 2. No. 1. pp. 1–18. [Available](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8?p2df) [Accesed: Apr. 2023].

* Egan J. VSB Power Line Fault Detection Approach. [Competition Notebook]. Kaggle: Power Line Fault Detection Competition of Enet Centre, VSB – TU of Ostrava. Egan J. 2019. [Available](https://www.kaggle.com/code/jeffreyegan/vsb-power-line-fault-detection-approach) [Accessed Sep. 2022].

* Higgins I, Matthey L, Pal A, Burgess C, Glorot X, Botvinick M, Mohamed S, Lerchner A. “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework”. International Conference on Learning Representations (ICLR). 2017. [Online]. [Available](https://openreview.net/forum?id=Sy2fzU9gl) [Accessed: Dec. 2021].

* Kingma DP, Welling M. “Auto-Encoding Variational Bayes”. arXiv:1312.6114v11 [stat.ML]. Dec 2013. [Online]. [Available](https://arxiv.org/abs/1312.6114) [Accessed: Dec. 2022].

* Mark4h. VSB_1st_place_solution. [Competition Notebook]. Kaggle: Power Line Fault Detection Competition of Enet Centre, VSB – TU of Ostrava. Mark4h. 2019. [Available](https://www.kaggle.com/code/mark4h/vsb-1st-place-solution) [Accessed Sep. 2022].
  
* Pytorch, Pytorch Tutorials (2022). Pytorch. Accessed: 2021 Dec 28 [Online]. [Available](https://pytorch.org/tutorials/) [Accessed: Apr. 2022].

* Pytorch, Pytorch Documentation (2022). Pytorch. Accessed: 2021 Dec 28 [Online]. [Available](https://pytorch.org/docs/stable/index.html) [Accessed: Apr. 2022].
