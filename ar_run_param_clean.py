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
    torch.cuda.set_device(3)
    datasetInputTensor = torch.Tensor(1, 1, 28, 28)
    #model = torch.load('./parent_models/mnist.pth')
    #teacherModel = torch.load('./parent_models/lenet_mnist.pth')
    teacherModel = torch.load('./parent_models/mnistvgg13.net')
    model = torch.load('stage1_mnist/995.net')
    baseline_acc = 0.995
    modelSavePath = './protos_mnist_param/'
else:
    print('Using cifar')
    import datasets.cifar10 as dataset
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    #teacherModel = torch.load('./parent_models/cifar10.pth')
    #teacherModel = torch.load('./parent_models/resnet18cifar.net')
    teacherModel = torch.load('./parent_models/cifar10_new.net')
    model = torch.load('stage1_vgg_cifar/9215.net')
    #model = torch.load('./results/c50lstm0_fc1.net')
    baseline_acc = 0.92
    modelSavePath = './protos_cifar_vgg_param/'

dataset.args.cuda = useCuda
parentSize = numParams(model)

constrained = False


def getEpsilon(iter, max_iter=15.0):
    return min(1, max(0, (1-iter/float(max_iter))**4)) #return 0

def getConstrainedReward(R_a, R_c, cons, vars, iter):
    eps = getEpsilon(iter)
    modelSize = vars[0]
    modelSizeConstraint = cons[0]
    if modelSize > modelSizeConstraint:
        return (eps - 1) + eps * (R_a * R_c)
    else:
        return R_a * R_c

def Reward(acc, params, baseline_acc, baseline_params, constrained=False, iter=50, cons=[], vars=[]):
    R_a = (acc/baseline_acc)
    C = (float(baseline_params - params))/baseline_params
    R_c = C*(2-C)
    if constrained:
        return getConstrainedReward(R_a, R_c, cons, vars, iter)
    return R_a * R_c

# Parameters for LSTM controller
num_layers = 2
num_hidden = 30
num_input = 5
num_output = 11
seq_len = 24

#lookup = [0.1, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
lookup = [0.25, .5, .5, .5, .5, .5, .6, .7, .8, .9, 1.0]

controller = LSTMAutoParams(num_input, num_output, num_hidden, num_layers, lookup)
opti = optim.Adam(controller.parameters(), lr=0.01)

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
    if m.fixed:
        return resizeToFit(Layer(m), inp)
    # Get representation
    # Perform updates
    _, k, s, o, p = Layer(m).getRepresentation()
    k = max(int(k * lookup[action[1]]), 1)
    #s = max(int(s * lookup[action[2]]), 1)
    o = max(int(o * lookup[action[3]]), 10)
    #p = max(int(p * lookup[action[4]]), 1)
    p = int(p*lookup[action[4]])
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

def traverse(parent, m, m_name, actions):
    global a
    global inp
    classname = m.__class__.__name__
    if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'Model']:
        child = createParentContainer(m)
        for i in m._modules.keys():
            if i == 'shortcut':
                continue
            #print(i)
            res = traverse(child, m._modules[i], i, actions)
            if res == None:
                return None
        if classname not in ['ResNet', 'VGG', 'LeNet', 'Model']:
            parent.add_module(m_name, child)
            return True
        else:
            return child
    else:
        if classname == 'Linear':
            inp = inp.view(inp.size(0), -1)
            #print(inp.size(1))
            if inp.size(1) > LINEAR_THRESHOLD:
                print('Linear layer too large')
                return None
        action = actions[a][:]
        m = applyActions(m, action, inp, lookup)
        if m == None:
            return None
        inp = m.cuda()(inp)
        parent.add_module(m_name, m)
        a += 1
    return True

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
    hashcode = hash(str(newModel)) if newModel != None else 0
    if hashcode in previousModels:
        R = previousModels[hashcode]
    elif newModel is None:
        R = -1
    else:
        print(newModel)
        acc = trainTeacherStudent(teacherModel, newModel, dataset)
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

def rollout_batch(model, N, e):
    global b
    global R_sum
    newModels = []
    idxs = []
    Rs = [0]*N
    actionSeqs = []
    studentModels = []
    for i in range(N):
        model_ = copy.deepcopy(model)
        layers = layersFromModule(model_)
        actions = rolloutActions(layers)
        actionSeqs.append(actions)
        newModel = build_child_model(model_, [a.data.numpy()[0] for a in actions])
        newModels.append(newModel)
        hashcode = hash(str(newModel)) if newModel else 0
        if hashcode in previousModels and constrained == False:
            Rs[i] = previousModels[hashcode]
        elif newModel is None:
            Rs[i] = -1
        else:
            studentModels.append(newModel)
            idxs.append(i)
    accs = trainTeacherStudentParallel(model, studentModels, dataset, epochs=5)
    print(accs)
    R = [Reward(accs[i], numParams(studentModels[i]), baseline_acc, parentSize) for i in range(len(accs))]
    for i in range(len(accs)):
        print(studentModels[i])
        torch.save(studentModels[i], modelSavePath + '%f.net' %(e+float(i)/10.0))
        comp = 1 - (numParams(studentModels[i])/float(parentSize))
        print('Compression ' + str(comp))
        print('Val accuracy ' + str(accs[i]))
        Rs[idxs[i]] = R[i]
    for i in range(len(Rs)):
        print('Reward achieved ' + str(Rs[i]))
    return (Rs, actionSeqs, newModels)
'''
def rollouts(N, model, e):
    Rs = []
    actionSeqs = []
    models = []
    for i in range(N):
        R, actions, newModel = rollout(copy.deepcopy(model), e + float(i)/10)
        Rs.append(R); actionSeqs.append(actions); models.append(newModel)
    return (Rs, actionSeqs, models)'''

def rollouts(N, model, e):
    Rs = []
    actionSeqs = []
    models = []
    (Rs, actionSeqs, models) = rollout_batch(copy.deepcopy(model), N, e)
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

