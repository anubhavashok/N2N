import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torchvision
import random
#from visualize import make_dot
from torch.nn.parameter import Parameter
from Model import Model
from utils import *
from Layer import Layer
import argparse
import copy
import signal
import sys
from controllers.AutoregressiveLayer import *


parser = argparse.ArgumentParser(description='Run layer only version')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='which dataset to test on')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()


datasetName = args.dataset
useCuda = args.cuda

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


def Reward(acc, params, baseline_acc, baseline_params):
    #R_acc = (baseline_loss - loss)^3 # determine what is chance as well
    R_acc = (acc/baseline_acc)
    C = (float(baseline_params - params))/baseline_params
    R_par = C*(2-C)
    #print('R_acc %f, R_par %f' % (R_acc, R_par))
    return R_acc * R_par
    # return R_acc*(R_par**2 + 0.3)

# Parameters for LSTM controller
num_layers = 2
num_hidden = 30
num_input = 5
num_output = 2
seq_len = 24


controller = LSTMAuto(num_input, num_output, num_hidden, num_layers)
opti = optim.Adam(controller.parameters(), lr=0.003)


# Store statistics for each model
accsPerModel = {}
paramsPerModel = {}
rewardsPerModel = {}
numSavedModels = 0

R_sum = 0
b = 0


'''
    Build child model
'''
def build_child_model(featureLayers, classifierLayers, actions):
    actions[0] = 1
    actions[len(featureLayers)] = 1
    featureActions = actions[:len(featureLayers)]
    classifierActions = actions[len(featureLayers):]
    classifierActions[-1] = 1
    
    featureLayers = [featureLayers[l] for l in range(len(featureLayers)) if featureActions[l]]
    classifierLayers = [classifierLayers[l] for l in range(len(classifierLayers)) if classifierActions[l]]
    
    features = nn.Sequential()
    classifier = nn.Sequential()
    
    inp = Variable(datasetInputTensor.clone().cuda())
    # Add first layer always to preserve input channels
    features.add_module('0', featureLayers[0]._layer)
    inp = featureLayers[0]._layer(inp)
    # Build feature sequence
    for i in range(1, len(featureLayers)):
        layer = resizeToFit(featureLayers[i], inp)
        features.add_module(str(i), layer)
        if featureLayers[i].type is 1: # Conv2d layer, add ReLU and BN
            features.add_module(str(i)+'b', nn.BatchNorm2d(layer.out_channels))
            features.add_module(str(i)+'r', nn.ReLU(inplace=False))
        inp = layer.cuda()(inp)
    
    numInputsToFC = inp.view(inp.size(0), -1).size(1)
    inp = inp.view(inp.size(0), -1)
    # Check if size is out of range
    if numInputsToFC < 10 or numInputsToFC > 30000:
        return None
    
    # Build classifier sequence
    for i in range(len(classifierLayers)):
        layer = resizeToFit(classifierLayers[i], inp)
        #classifier.add_module('cd%d' % i, nn.Dropout())
        classifier.add_module('c%d' % i, layer)
        if i != (len(classifierLayers)-1):
            classifier.add_module('cr%d' % i, nn.ReLU(inplace=False))
        inp = layer.cuda()(inp)
    
    # Build whole model
    newModel = Model(features, classifier)
    newModel = resetModel(newModel)
    # Check if any compression has been achieved
    if numParams(newModel) > parentSize:
        return None
    
    return newModel


def rolloutActions(featureLayers, classifierLayers):
    global controller
    hn = Variable(torch.zeros(1, num_hidden))
    cn = Variable(torch.zeros(1, num_hidden))
    input = Variable(torch.Tensor(len(featureLayers) + len(classifierLayers), 1, num_input))
    for i in range(len(featureLayers)):
        input[i] = featureLayers[i].toTorchTensor()
    for i in range(len(classifierLayers)):
        input[i + len(featureLayers)] = classifierLayers[i].toTorchTensor()
    print(input.size())
    output = controller(input, (hn, cn))
    probs = probs.squeeze(1)
    actions = probs.multinomial()
    return actions 


def rollout(model_, i):
    global b
    global R_sum
    featureLayers = [Layer(l) for l in model_.features._modules.values()]
    featureLayers = list(filter(lambda x: x.type in [1, 2, 8], featureLayers))
    classifierLayers = [Layer(l) for l in model_.classifier._modules.values()]
    classifierLayers = list(filter(lambda x: x.type in [5], classifierLayers))
    actions = rolloutActions(featureLayers, classifierLayers)
    newModel = build_child_model(featureLayers, classifierLayers, [a.data.numpy()[0] for a in actions])
    if newModel is None:
        R = -1
    else:
        print(newModel)
        acc = train(dataset, newModel)
        R = Reward(acc, numParams(newModel), baseline_acc, parentSize)
        rewardsPerModel[i] = R
        accsPerModel[i] = acc
        paramsPerModel[i] = numParams(newModel)
        #torch.save(newModel, modelSavePath + '%f.net' % i)
        #print('Val accuracy: %f' % acc)
        print('Compression: %f' % (1.0 - (float(numParams(newModel))/parentSize)))
        print('Reward achieved %f' % R)
        print('Reward after baseline %f' % (R-b))
        # Update reward and baseline after each rollout
    return (R, actions, newModel)

def rollouts(N, model, e):
    Rs = []
    actionSeqs = []
    models = []
    for i in range(N):
        R, actions, newModel = rollout(copy.deepcopy(model), e + float(i)/10)
        Rs.append(R); actionSeqs.append(actions); models.append(newModel)
    return (Rs, actionSeqs, models)

def update_controller(actionSeqs, avgR):
    print('Reinforcing for epoch %d' % e)
    for actions in actionSeqs:
        print(actions.size())
        actions.reinforce(avgR - b)
        opti.zero_grad()
        autograd.backward(actions, [None for _ in actions])
        opti.step()

epochs = 30
N = 5
for e in range(epochs):
    # Compute N rollouts
    (Rs, actionSeqs, models) = rollouts(N, model, e)
    # Compute average reward
    avgR = np.mean(Rs)
    print('Average reward: %f' % avgR)
    b = R_sum/float(e+1)
    R_sum = R_sum + avgR
    # Update controller
    update_controller(actionSeqs, avgR)
resultsFile = open(modelSavePath + 'results.txt', "w")
output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel)
