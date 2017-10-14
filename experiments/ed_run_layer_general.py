import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torchvision
import random
from visualize import make_dot
from torch.nn.parameter import Parameter
from Model import Model
from utils import *
from Layer import Layer
import argparse
import copy
from controllers.EncoderDecoder import * 



import signal
import sys

parser = argparse.ArgumentParser(description='Run layer only version')
parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                    help='which dataset to test on')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()


datasetName = args.dataset
useCuda = args.cuda

# Define dataset variables
# We need:
# 1. size of input data
# 2. pre-trained Parent model
# 3. baseline accuracy of parent model, hardcode since we don't want to run everytime we load
datasetInputTensor = None
baseline_acc = 0
modelSavePath = None
if datasetName is 'mnist':
    print('Using mnist')
    import datasets.mnist as dataset
    datasetInputTensor = torch.Tensor(1, 1, 28, 28)
    model = torch.load('./parent_models/mnist.pth')
    baseline_acc = 0.994
    modelSavePath = './protos_mnist/'
else:
    print('Using cifar')
    import datasets.cifar10 as dataset
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('./parent_models/cifar10.pth')
    baseline_acc = 0.88
    modelSavePath = './protos_cifar/'

dataset.args.cuda = useCuda
parentSize = numParams(model)

# Define Observation space


# Define Action space


# Define Loss
def Reward(acc, params, baseline_acc, baseline_params):
    #R_acc = (baseline_loss - loss)^3 # determine what is chance as well
    R_acc = (acc/baseline_acc)
    C = (float(baseline_params - params))/baseline_params
    R_par = C*(2-C)
    print('R_acc %f, R_par %f' % (R_acc, R_par))
    return R_acc * R_par
    # return R_acc*(R_par**2 + 0.3)


layers = [Layer(l) for l in model.features._modules.values()]
layers = list(filter(lambda x: x.type in [1, 2], layers))
    

# Parameters for LSTM
num_layers = 2
num_hidden = 30
num_input = 5
num_output = 2
seq_len = len(layers) 

# LSTM definition
#lstm = nn.LSTM(num_input, num_hidden, num_layers)
#Wt_softmax = nn.Linear(num_hidden, num_output)
#softmax = nn.Softmax()
lstm = EncoderDecoderLSTM(num_input, num_output, num_hidden, seq_len)

# Optimizer definition
opti = optim.Adam(lstm.parameters(), lr=0.0006)

epochs = 100
R_sum = 0
lookup = [0, 1]

print('About to setup models')

# ReLU fix
# layers = [l if l.type is not 'ReLU' else Layer(nn.ReLU(inplace=False)) for l in layers]

# Store statistics for each model
accsPerModel = {}
paramsPerModel = {}
prevMaxReward = 0
numSavedModels = 0

def signal_handler(signal, frame):
        print('Ending experiment')
        resultsFile = open(modelSavePath + 'results.txt', "w")
        output_results(resultsFile, accsPerModel, paramsPerModel)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



for e in range(epochs):
    # Create layers
    model_ = copy.deepcopy(model)
    layers = [Layer(l) for l in model_.features._modules.values()]
    layers = list(filter(lambda x: x.type in [1, 2], layers))
    classifierLayers = [Layer(l) for l in model_.classifier._modules.values()]
    print len(layers)
    
    input = torch.Tensor(len(layers), 1, num_input)
    for i in range(len(layers)):
        input[i] = layers[i].toTorchTensor().squeeze(0)
    input = Variable(input)
    outputs = lstm.forward(input)
    actions = [o.multinomial() for o in outputs]
    intActions = [a.data.numpy()[0] for a in actions]
    # Generate new model
    features = nn.Sequential()
    input = Variable(datasetInputTensor)
    intActions[0] = 1 # Save first layer so that input size is consistent
    for i in [i for i, x in enumerate(intActions) if x == 1]:
        if 'weight' in layers[i]._layer._parameters:
            in_channels = input.size()[1]
            _, kernel_size, stride, out_channels, padding = layers[i].getRepresentation()
            layers[i] = Layer(resizeLayer(layers[i]._layer, in_channels, out_channels, kernel_size, stride, padding))
        features.add_module(str(i), layers[i]._layer)
        if layers[i].type == 1:
            features.add_module(str(i)+'b', nn.BatchNorm2d(layers[i]._layer.out_channels))
            features.add_module(str(i)+'r', nn.ReLU(inplace=False))
        input = layers[i]._layer.forward(input)
        if input.size(2) * input.size(3) < 5:
            break
    # Check a few things to disqualify the model
    # 1. Representation size
    fc_in_channels = input.size(1)*input.size(2)*input.size(3)
    print 'Current fc nodes', fc_in_channels
    classifier = nn.Sequential()

    if fc_in_channels < 10 or fc_in_channels > 100000:
        R = -1
    else:
        # Set fc in channels
        modules = classifierLayers
        print('Setting classifier layers')
        modules[1] = Layer(resizeLayer(modules[1]._layer, fc_in_channels, modules[1]._layer.out_features))
        print('Resized fc layer')
        for j in range(len(modules)):
            classifier.add_module('c%d' % j, modules[j]._layer)
        # Attach classifier
        # Run model and determine accuracy
        newModel = Model(features, classifier)
        newModel = resetModel(newModel)
        print(newModel)
        if numParams(newModel) > parentSize:
            R = -1
        else:
            newModel = newModel.cuda() if dataset.args.cuda else newModel
            acc = train(dataset, newModel)
            #viz = make_dot(output)
            #viz.render('./vis/%d' % e)
            accsPerModel[e] = acc
            paramsPerModel[e] = numParams(newModel)
            R = Reward(acc, numParams(newModel), baseline_acc, numParams(model))
            print('Val accuracy: %f' % acc)
            print('Compression: %f' % (1.0 - (float(numParams(newModel))/numParams(model))))
            print('Reward achieved %f' % R)
            if R >= prevMaxReward and numSavedModels < 25:
                torch.save(newModel, modelSavePath + '%d.net' % e)
                numSavedModels += 1
            prevMaxReward = max(prevMaxReward, R)
            b = R_sum/float(e+1)
            R_sum = R_sum + R
    actions.reverse()
    for k in range(len(actions)):
        actions[k].reinforce(R-b)
    print('Reinforcing for epoch %d' % e)
    opti.zero_grad()
    autograd.backward(actions, [None for _ in actions])
    opti.step()

# Print statstics
resultsFile = open(modelSavePath + 'results.txt', "w")
output_results(resultsFile, accsPerModel, paramsPerModel)
