# Leaning Persistent Poincare Representations
This repository contains the code and instructions to run the experiments and replicate the results reported in the paper titled 'Learning Persistent Poincare Representations' submitted to NeurIPS 2020.

# Requirements 
To run the provided files the following requirements must be met:

* Python (>= 3.7)
* TensorFlow (>= 2.0) 
* TensorBoard (>= 2.2)
* Numpy (>= 1.18.4)
* Giotto-TDA (0.2.1)
* SciPy (>= 0.17.0)
* Scikit-learn (>= 0.22.0)
* Matplotlib (>= 3.0.3)
* Networkx (>= 2.4)
* Ceckmate (>= 0.0.7)
* Tqdm (>=4.46.0)

They can all be installed with the following pip command:
```
pip install tensorflow-gpu tensorboard numpy giotto-tda scipy matplotlib networkx cechmate tqdm
```
It is recommenced to use the GPU version of TensorFlow so that the experiments are completed within a reasonable time frame.
# Datasets
All the image datasets will be automatically downloaded through the Keras backend. The graph datasets can be downloaded from this [link](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets). Please download the datasets mentioned bellow and extract them in the subdirectory ``datasets`` without modifying the directory names. 
Note that the script works for any unlabeled (i.e., no node or edge labels) graph dataset from the above link.
However, it has been tested on the ones mentioned bellow. 
# Run  
The starting file to run the experiments is the ``main.py``. It can be called as follows:

```
main.py [-h] [-m MAN_DIM] [-K PROJ_BASES] [-s SPACES] [-e EPOCHS]  [-b BATCH_SIZE] [-g GPUS]
               data_type data_id 
```
* data_type: 'images' or 'graphs' (required) 
* data_id: unique id for each dataset (required)
    * images: 'mnist' or 'fashion-mnist'
    * graphs: 'IMDB_BINARY' or 'IMDB-MULTI' or 'REDDIT_BINARY' or 'REDDIT-MULTI-5K' or 'REDDIT-MULTI-12K'
* MAN_DIM: the dimension of the Poincare ball, if not given it will iterate over all m=3,6,9,10
* PROJ_BASES: how many projection bases to explore, it is recommended to try small value for images (5-10) and large values (200-500) for graphs
* EPOCHS, BATCH_SIZE: self-explanatory
* GPUS: data parallelism over given gpus, if set to -1 it will use all gpus

The training logs are saved in the ``logs/fit`` directory with the folder name: ``{data_id}_K{PROJ_BASES}_m{MAN_DIM}_s{space}``. To view them in TensorBoard run the following command:
```
tensorboard --logdir logs
```
The above hyper-parameters are also saved using the HPARAMS module of TensorBoard so that they can be easily isolated and the relevant loss/acc plots are more clean. 
# Baselines
The baseline results reported on Table 2 are taken from the respective papers. Regarding Table 2, we obtained the reported results by using the respective authors' publically available code and training the model using the same persistence diagrams as those used to obtain our results. We attach the code and instructions to run the baselines.
* ``baselines/kusano`` contains the code for the Persistent Weighted Gaussian Kernel by Kusano et al. Taken from [GitHub](https://github.com/genki-kusano/gram_matrix/tree/master/). Baseline can be run using the ``baseline_pwgk`` notebook.
* ``baselines/adam`` contains the code for the Persistent Image embedding by Adams et el. Baseline can be run using the ``baseline_pi`` notebook.
* ``baselines/hofer`` contains the code for Lernable Representation presented by Hofer et al. Taken from [GitHub](https://github.com/c-hofer/jmlr_2019). Baseline can be run using the ``baseline_glr`` notebook.