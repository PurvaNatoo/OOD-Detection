# OOD-Detection

Machine Learning systems are often built and trained with a closed-world assumption i.e. the distribution of test data is assumed to be similar to the distribution of training data. However, in real-world scenarios, this assumption does not necessarily hold true, and this might cause the system to produce highly confident results on Out-of-distribution (OOD) data. To ensure the successful and safe deployment of machine learning systems, especially in intolerant domains like healthcare, the systems must be able to distinguish between data that is significantly different from the training data. In this code, we have presented a comparative study of the different OOD detection techniques.

### Project details
In-distribution Dataset: CIFAR10
Out of Distribution Dataset: Textures(dtd), LSUN, iSUN, WaRP, and Oxford Flowers 102
Pretrained ML model: Resnet18
OOD methods compared: MSP, ODIN, Energy based OOD detection, ReAct[1], KNN[2]

### Project Setup

1. Download the KNN and ReAct folders. 
2. The datasets and data folder are important and need some attention. I have added a readme file named "Datasets_info" in the datasets and data folders which explains which datasets should be downlaoded to that folder and also provides a link to that dataset. The data and datasets folders are present in both KNN and ReAct folders.
3. To run the KNN code use these commands: (change the out-datasets parameter based on the dataset you are using)
```
python feat_extract.py --in-dataset CIFAR-10  --out-datasets oxford_flowers --name resnet18  --model-arch resnet18
python run_cifar.py --in-dataset CIFAR-10  --out-datasets oxford_flowers --name resnet18  --model-arch resnet18
```
OR 
You can change the parameters in args_loader.py and use these commands:
```
python feat_extract.py
python run_cifar.py
```
4. To run the ReAct code use this command: (You can change the parameters in args_loader.py)
```
python eval.py
```


Note: This code is a modification of the code taken from https://github.com/deeplearning-wisc/knn-ood and https://github.com/deeplearning-wisc/react
This repository is to help people setup the code for Small Scale Experiment.

### References

[1] Sun, Y., Guo, C., and Li, Y., “ReAct: Out-of-distribution Detection With Rectified Activations”, arXiv e-prints, 2021. doi:10.48550/arXiv.2111.12797.\
[2] Sun, Y., Ming, Y., Zhu, X., and Li, Y., “Out-of-Distribution Detection with Deep Nearest Neighbors”, arXiv e-prints, 2022. doi:10.48550/arXiv.2204.06507.

