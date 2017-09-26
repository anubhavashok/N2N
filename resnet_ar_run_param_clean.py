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
from controllers.AutoregressiveParam import *


parser = argparse.ArgumentParser(description='Run layer only version')
parser.add_argument('--dataset', type=str, default='svhn', metavar='N',
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
    #model = torch.load('./parent_models/mnist.pth')
    teacherModel = torch.load('./parent_models/lenet_mnist.pth')
    model = torch.load('stage1_mnist/lenet_layer_reduced.net')
    #baseline_acc = 0.989
    baseline_acc = 0.987
    modelSavePath = './protos_mnist/'
elif datasetName is 'caltech256':
    print('Using caltech256')
    import datasets.caltech256 as dataset
    datasetInputTensor = torch.Tensor(1, 3, 224, 224)
    teacherModel = torch.load('./caltech256_resnet18.net') 
    model = torch.load('./stage1_caltech256/best.net')
    baseline_acc = 0.58
    modelSavePath = './protos_caltech256_param'
elif datasetName is 'cifar100':
    print('Using cifar100')
    torch.cuda.set_device(1)
    import datasets.cifar100 as dataset
    baseline_acc = 0.72
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('stage1_cifar100_resnet34/best.net')
    teacherModel = torch.load('./parent_models/resnet34_cifar100.net')
    modelSavePath = './protos_cifar100_stage_2/'
elif datasetName is 'svhn':
    print('Using svhn')
    import datasets.svhn as dataset
    baseline_acc = 0.94
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    teacherModel = torch.load('./parent_models/svhn_resnet18.net')
    model = torch.load('./stage1_svhn/95.66.net')
    modelSavePath = './protos_svhn_stage2/'
else:
    print('Using cifar')
    import datasets.cifar10 as dataset
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('stage1_resnet34/93_fixed.net')
    teacherModel = torch.load('./parent_models/resnet18_best.net')
    baseline_acc = 0.912
    modelSavePath = './protos_cifar10_resnet18_stage2/'

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
num_output = 11
seq_len = 24

#lookup = [0.1, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
lookup = [0.25, .5, .5, .5, .5, .5, .6, .7, .8, .9, 1.0]

controller = LSTMAutoParams(num_input, num_output, num_hidden, num_layers, lookup)
opti = optim.Adam(controller.parameters(), lr=0.1)

previousModels = {}

# Store statistics for each model
accsPerModel = {}
paramsPerModel = {}
rewardsPerModel = {}
numSavedModels = 0

R_sum = 0
b = 0

LINEAR_THRESHOLD = 50000


def applyActions(m, action, inp, lookup):
    #if m.fixed:
    #    return resizeToFit(Layer(m), inp)
    # Get representation
    # Perform updates
    _, k, s, o, p = Layer(m).getRepresentation()
    k = max(int(k * lookup[action[1]]), 1) if m.fixed[1] == False else k
    s = max(int(s * lookup[action[2]]), 1) if m.fixed[2] == False else s
    o = max(int(o * lookup[action[3]]), 10) if m.fixed[3] == False else o
    p = int(p * lookup[action[4]]) if m.fixed[4] == False else p
    in_channels = inp.size(1)
    cn = m.__class__.__name__
    if cn == 'Linear':
        in_channels = inp.view(inp.size(0), -1).size(1)
        if in_channels > LINEAR_THRESHOLD or in_channels < 10:
            print('Linear layer too large')
            return None
    return resizeLayer(m, in_channels, o, kernel_size=k, stride=s, padding=p)

a = 0
inp = Variable(datasetInputTensor.clone()).cuda()
def processBlock(actions, m, lookup, input_size):
    global a
    finalAction = actions[a+len(m.layers._modules)-1][3]
    finalActionUsed = False
    
    secondFinalAction = actions[a+len(m.layers._modules)-2][3]
    secondFinalActionUsed = False
    
    firstConv = False
    secondConv = False
    hasShortcut = False
    
    if '0' in m.layers._modules:
        firstConv = True
    
    if '3' in m.layers._modules:
        secondConv = True
    
    if hasattr(m, 'shortcut') and m.shortcut != None:
        hasShortcut = True
    o = input_size 
    if firstConv:
        i = input_size#m.layers._modules['0'].in_channels
        k = m.layers._modules['0'].kernel_size
        s = m.layers._modules['0'].stride
        o = m.layers._modules['0'].out_channels
        if secondConv:
            o = max(int(o * lookup[finalAction]), 10)
            finalActionUsed = True
        elif hasShortcut:
            o = max(int(o * lookup[finalAction]), 10)
            si = i
            sk = m.shortcut._modules['0'].kernel_size
            ss = m.shortcut._modules['0'].stride
            sp = m.shortcut._modules['0'].padding
            m.shortcut._modules['0'] = resizeLayer(m.shortcut._modules['0'], si, o, sk, ss, sp).cuda()
            m.shortcut._modules['1'] = resizeLayer(m.shortcut._modules['1'], o, o).cuda()
            finalActionUsed = True
        else:
            # We want output to be same as input in the event of no shortcut and no secondConv
            o = i 
        p = m.layers._modules['0'].padding
        m.layers._modules['0'] = resizeLayer(m.layers._modules['0'], i, o, k, s, p).cuda()
    if '1' in m.layers._modules:
        m.layers._modules['1'] = resizeLayer(m.layers._modules['1'], o, o).cuda()
    if secondConv:
        #i = m.layers._modules['3'].in_channels if not firstConv else m.layers._modules['0'].out_channels
        i = o
        k = m.layers._modules['3'].kernel_size
        s = m.layers._modules['3'].stride
        o = m.layers._modules['3'].out_channels
        if hasShortcut:
            o = max(int(o * lookup[secondFinalAction]), 10)
            si = m.layers._modules['0'].in_channels if firstConv else i
            sk = m.shortcut._modules['0'].kernel_size
            ss = m.shortcut._modules['0'].stride
            sp = m.shortcut._modules['0'].padding
            m.shortcut._modules['0'] = resizeLayer(m.shortcut._modules['0'], si, o, sk, ss, sp).cuda()
            m.shortcut._modules['1'] = resizeLayer(m.shortcut._modules['1'], o, o).cuda()
            secondFinalActionUsed = True
        else:
            o = m.layers._modules['0'].in_channels if firstConv else i
        p = m.layers._modules['3'].padding
        m.layers._modules['3'] = resizeLayer(m.layers._modules['3'], i, o, k, s, p).cuda()
    if '4' in m.layers._modules:
        m.layers._modules['4'] = resizeLayer(m.layers._modules['4'], o, o).cuda()
    
    # Void actions
    for a in range(len(m.layers._modules)-2):
    #    actions[a].detach()
        a += 1
    
    #if not secondFinalActionUsed:
    #    actions[a].detach()
    a += 1
    
    #if not finalActionUsed:
    #    actions[a].detach()
    a += 1
    return m

def traverse(parent, m, m_name, actions):
    global a
    global inp
    classname = m.__class__.__name__
    if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'Model', 'ResNetModifiable', 'BasicBlockModifiable']:
        # Change the number of input channels of the first conv of the shortcut layer
        oldInp = Variable(copy.deepcopy(inp.data))
        child = createParentContainer(m)
        if classname in ['BasicBlock', 'BottleNeck', 'BasicBlockModifiable']:
            fixBlockLayers(m)
            m = processBlock(actions, m, lookup, inp.size(1)).cuda()
            inp = m.layers(inp.cuda())
            child = m 
        else:
            for i in m._modules.keys():
                res = traverse(child, m._modules[i], i, actions)
                if res == None:
                    return None
        # Change the number of output channels of the last conv of the shortcut layer
        if classname not in ['ResNet', 'VGG', 'LeNet', 'Model', 'ResNetModifiable']:
            child(oldInp)
            parent.add_module(m_name, child)
            return True
        else:
            return child
    else:
        if classname == 'Linear':
            inp = inp.view(inp.size(0), -1)
            #print(inp.size(1))
            if inp.size(1) > LINEAR_THRESHOLD or inp.size(1) < 10:
                print('Linear layer too large')
                return None
        action = actions[a][:]
        m = applyActions(m, action, inp, lookup)
        if m == None:
            return None
        try:
            inp = m.cuda()(inp)
        except:
            print('Error in model, probably because of receptive field size')
            return None
        parent.add_module(m_name, m)
        a += 1
    return True
'''
def fixLayers(m):
    layers = flattenModule(m)
    # Initialize
    for l in layers:
        l.fixed = False
    
    # Fix any layers you want here
    # ----
    # Fix final linear layer
    layers[1].fixed = True
    layers[2].fixed = True
    layers[-1].fixed = True
    layers[-2].fixed = True
    # ----
'''
def fixBlockLayers(m):
    # Only allow num_filters of conv layers to change
    for mm in m.layers._modules.values():
        mm.fixed = [True]*5
    m.layers._modules.values()[0].fixed = [True, True, True, False, True]
    #m._modules.values()[-2].fixed = [True, True, True, False, True]
    

def fixLayers(m):
    layers = flattenModule(m)
    # Initialize
    for l in layers:
        l.fixed = [False]*5

    # Fix any layers you want here
    # ----
    #     Fix all shortcut layers and corresponding stride layers, but not pre layers
    for l in layers:
        # Fix all shortcut/downsampling layers
        # Since we couple the action for the conv layer and this layer we can modify this when building model
        cn = l.__class__.__name__
        if hasattr(l, 'stride') and l.stride != (1, 1) and cn == 'Conv2d':
            l.fixed = [True]*5
        # Fix final linear and average pooling layer
        if cn == 'Linear' or cn == 'AvgPool2d':
            l.fixed = [True]*5
    # ----

'''
    Build child model
'''
def build_child_model(m, actions):
    # We eliminate a layer if any one of the coefficients are = 0
    global inp
    global a
    a = 0
    actions = np.reshape(actions, (-1, num_input))
    
    inp = Variable(datasetInputTensor.clone()).cuda()
    fixLayers(m)
     
    # Build whole model
    newModel = traverse(None, m, None, actions)
    if newModel == None:
        print('newModel is none for some reason')
        return None
    resetModel(newModel)
    # Check if any compression has been achieved
    if numParams(newModel) > parentSize:
        print('newModel is larger than parent')
        return None
     
    return newModel


def rolloutActions(layers):
    global controller
    hn = [Variable(torch.zeros(1, num_hidden))] * num_layers
    cn = [Variable(torch.zeros(1, num_hidden))] * num_layers
    input = Variable(torch.Tensor(len(layers), 1, num_input))
    for i in range(len(layers)):
        input[i] = Layer(layers[i]).toTorchTensor(skipSupport=False)
    output = controller(input, (hn, cn))
    return output


def rollout(model_, e):
    global b
    global R_sum
    layers = layersFromModule(model_)
    actions = rolloutActions(layers)
    newModel = build_child_model(model_, [a.data.numpy()[0] for a in actions])
    actionsMask = np.ravel([l.fixed for l in layers])
    newActions = []
    for i in range(len(actionsMask)):
        if actionsMask[i]:
            newActions.append(actions[i])
    actions = newActions
    print(newModel)
    hashcode = hash(str(newModel)) if newModel != None else 0
    print(hashcode)
    if hashcode in previousModels:
        R = previousModels[hashcode]
    elif newModel is None:
        R = -1
    else:
        print(newModel)
        acc = trainTeacherStudent(teacherModel, newModel, dataset, epochs=5) if datasetName is not 'caltech256' else trainNormal(newModel, dataset, epochs=3)
        R = Reward(acc, numParams(newModel), baseline_acc, parentSize)
        previousModels[hashcode] = R
        rewardsPerModel[i] = R
        accsPerModel[i] = acc
        paramsPerModel[i] = numParams(newModel)
        torch.save(newModel, modelSavePath + '%f.net' % e)
        print('Val accuracy: %f' % acc)
        print('Compression: %f' % (1.0 - (float(numParams(newModel))/parentSize)))
        print('Reward achieved %f' % R)
        #print('Reward after baseline %f' % (R-b))
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
        for action in actions:
            action.reinforce(avgR - b)
        opti.zero_grad()
        autograd.backward(actions, [None for _ in actions])
        opti.step()

epochs = 50
N = 3
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

