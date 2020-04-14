# General Clustering Methodology
+ A template for general clustering method, can make interface of different data type, including images

## Version:
+ V 1.0

## Model included:
+ KMeans


## Preparation
+ Requirements:
    + Python >= 3.6 & < 3.8
    + sklearn = 0.22.1


## Training Steps
+ Data Input Preparation:
    + Can be parametric data or image data, but need to preprocess to array format
    + If you put image data, please directly put the data under dataset

```
|——dataset
   |——data_1.jpg
   |——data_2.jpg
   ...
   |——data_x.jpg

```

+ Configurations
    + Change the corresponding parameters in **config.py**.


## Train/Evaluation Steps
+ Run **python train.py** in shell/ps/cmd to evaluate the model's performance on the test dataset
