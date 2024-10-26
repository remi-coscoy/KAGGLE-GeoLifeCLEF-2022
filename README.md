# GeoLifeCLEF 2022 Kaggle Challenge

The aim GeoLifeCLEF is to predict the localization of plant and animal species based on satellite images and environmental features.

We present here several approaches to this challenge:

Convolutional Neural Network

- First benchmark model, poor performance

Visual Transformer

Multimodal CNN

- Using both images and tabular data in the same model

InceptionV3

Random Forest

- Calculation of new, more relevant features from the tabular data

# Links

Link to the kaggle competition :
https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9

And Leaderboard for possible solutions :
https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/leaderboard

Link to PyTorch template code :
https://github.com/jeremyfix/pytorch_template_code/

# Installation

This code is configured to run on a cluster of gpus, please change the submit mechanism as required.

## To Train

To train the model, you can use the following command:

- CNN :

```bash## 
python3 submit-slurm.py config.yaml 1
```

- Random Forest : (use n_samples = 0 to use all samples)

```bash
python3 slurm-rf.py [n_samples] [n_estimators] [max_depth]
```

## To make a submission

To make a submission, you can use the following command:

```bash
python3 eval-model-slurm.py VanillaCNN_0
```

Here VanillaCNN_0 is the name of the model you want to evaluate. It should be a folder in the logs folder, containing a `best_model.pt` file.

