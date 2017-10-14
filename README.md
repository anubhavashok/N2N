# N2N: Network to Network Compression using Policy Gradient Reinforcement Learning

[Link to ArXiV paper](https://arxiv.org/abs/1709.06030)

## How to run
Run using run.py.
Specify the pre-trained model and dataset to use as arguments.
Detailed instructions can be found in the help menu in run.py
### Pre-requisites
There are some pre-requisites for running this
1. python >= 2.7
2. pytorch >= 0.2
3. torchvision >= 0.19

### Removal
Here is an example command to train the layer removal policy on the cifar10 dataset using the resnet-18 model
```
python run.py removal cifar10 teacherModels/resnet18_cifar10.net --cuda True 
```

### Shrinkage
NOTE: To run shrinkage, specify both teacher model and reduced model from stage1
```
python run.py shrinkage cifar10 teacherModels/resnet18_cifar10.net --model Stage1_cifar10/reduced_model1.net --cuda True 
```

## Downloading models
All models can be downloaded at this [link](https://cmu.box.com/s/v15vgm6mg9ryemkazkrgew6rjovctstv)
### Pre-trained teacher models
The teacher models are to be specified to run.py to train.
### Pre-trained student models
The pre-trained student models are given to show the performance of the models described in the paper. They can be tested using test\_model.py
Test using 
```
python test_model.py studentModels/resnet18_cifar10.net cifar10
```
### Pre-trained policies
The pre-trained polcies are specified to run the transfer learning experiments



## Experiments folder
The experiments folder contains various variants of layer removal and shrinkage that were tried for the actual paper. These were mainly experiments which require substantial modifications to the main code or were used on earlier iterations of the project
They have to be moved to the main folder before being run.
The following describes each experiment
1. ar\_run\_layer\_clean.py - Layer removal using the Autoregressive controller
2. ar\_run\_param\_clean.py - Layer shrinkage for **Non-ResNet** convolutional models
3. bd\_run\_layer\_clean.py - Layer removal for **Non-ResNet** convolutional models using the bidirectional controller
4. ed\_run\_layer\_general.py - Layer removal for **Non-ResNet** convolutional models using the encoder-decoder controller
5. resnet\_actor\_critic\_layer.py - Layer removal using the Actor-Critic controller
6. resnet\_ar\_run\_layer\_clean.py - Layer removal for **ResNet** models using the Autoregressive controller
