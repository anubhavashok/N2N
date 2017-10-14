# N2N: Network to Network Compression using Policy Gradient Reinforcement Learning

[Link to ArXiV paper](https://arxiv.org/abs/1709.06030)

## How to run
Run using run.py.
Specify the pre-trained model and dataset to use as arguments.

### Removal
Here is an example command to train the layer removal policy on the cifar10 dataset using the resnet-18 model
```
python run.py removal cifar10 teacherModels/resnet18_cifar10.net --cuda True 
```

### Shrinkage
```
python run.py shrinkage cifar10 teacherModels/resnet18_cifar10.net --model Stage1_cifar10/reduced_model1.net --cuda True 
```

## Downloading models
The teacher models are to be specified to run.py to train.
The pre-trained student models are given to show the performance of the models described in the paper. They can be tested using test_model.py
The pre-trained polcies are specified to run the transfer learning experiments
### Pre-trained teacher models
### Pre-trained student models
### Pre-trained policies
