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
from model import resnet
from controllers.ActorCriticLSTM import LSTM

sys.path.insert(0, '/home/anubhava/ssd.pytorch/')

constrained = False#True

parser = argparse.ArgumentParser(description='Run layer only version')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='which dataset to test on')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
args = parser.parse_args()


datasetName = args.dataset
useCuda = args.cuda
loadController = False 
skipSupport = False 

datasetInputTensor = None
baseline_acc = 0
modelSavePath = None
controllerSavePath = None
if datasetName is 'mnist':
    print('Using mnist')
    import datasets.mnist as dataset
    datasetInputTensor = torch.Tensor(1, 1, 28, 28)
    model = torch.load('./parent_models/mnistvgg13.net')
    #model = torch.load('./parent_models/lenet_mnist.pth')
    baseline_acc = 0.9955
    #baseline_acc = 0.983
    modelSavePath = './protos_mnist/'
    controllerSavePath = './controllers_mnist/lstm_lenet.net'
    controllerLoadPath = './controllers_mnist/lstm_lenet.net'
elif datasetName is 'cifar100':
    print('Using cifar100')
    import datasets.cifar100 as dataset
    baseline_acc = 0.67
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    model = torch.load('./parent_models/vgg19cifar100.net')
    modelSavePath = './protos_cifar100/'
    controllerSavePath = './controllers_cifar100/lstm_resnet18.net'
    controllerLoadPath = './controllers_cifar100/lstm_resnet18.net'
elif datasetName is 'caltech256':
    print('Using caltech256')
    import datasets.caltech256 as dataset
    baseline_acc = 0.79
    datasetInputTensor = torch.Tensor(1, 3, 224, 224)
    model = torch.load('./parent_models/caltech256_resnet18.net')
    modelSavePath = './protos_caltech256/'
    controllerSavePath = './controllers_caltech256/lstm_resnet18.net'
    controllerLoadPath = './controllers_caltech256/lstm_resnet18.net'
elif datasetName is 'imagenet':
    print('Using imagenet')
    import datasets.imagenet as dataset
    torch.cuda.set_device(2)
    datasetInputTensor = torch.Tensor(1, 3, 224, 224)
    model = torch.load('./parent_models/resnet34_imagenet.net')
    modelSavePath = './protos_imagenet'
    controllerSavePath = './controllers_imagenet/lstm_resnet34.net'
    controllerLoadPath = './controllers_imagenet/lstm_resnet34.net'
else:
    print('Using cifar')
    torch.cuda.set_device(0)
    import datasets.cifar10_old as dataset
    datasetInputTensor = torch.Tensor(1, 3, 32, 32)
    #model = torch.load('./parent_models/cifar10.pth')
    #baseline_acc = 0.88
    model = torch.load('./parent_models/resnet18cifar.net')
    #model = torch.load('./parent_models/resnet34cifar.net')
    #model = torch.load('./parent_models/cifar_vgg19.net')
    baseline_acc = 0.9205
    modelSavePath = './protos_cifar/'
    #controllerSavePath = './controllers_cifar/lstm_vgg19.net'
    #controllerSavePath = './controllers_cifar/lstm_resnet34.net'
    controllerSavePath = './controllers_cifar/lstm_resnet18.net'
    controllerLoadPath = './controllers_cifar/lstm_resnet34.net'

dataset.args.cuda = useCuda
parentSize = numParams(model)

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
    R_a = (acc/baseline_acc) #if acc > 0.92 else -1
    C = (float(baseline_params - params))/baseline_params
    R_c = C*(2-C)
    if constrained:
        return getConstrainedReward(R_a, R_c, cons, vars, iter)
    return R_a * R_c

# Parameters for LSTM controller
num_layers = 2
num_hidden = 30
num_input = 7 if skipSupport else 5
num_output = 2
seq_len = 1

controller = LSTM(num_input, num_output, num_hidden, num_layers, bidirectional=True)
if loadController:
    controller = torch.load(controllerLoadPath)
opti = optim.Adam(controller.parameters(), lr=0.003)

previousModels = {}
# Store statistics for each model
accsPerModel = {}
paramsPerModel = {}
rewardsPerModel = {}
numSavedModels = 0

R_sum = 0
b = 0

LINEAR_THRESHOLD = 50000

a = 0
inp = Variable(datasetInputTensor.clone()).cuda()
def traverse(parent, m, m_name, actions):
    global a
    global inp
    classname = m.__class__.__name__
    if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model']:
        child = createParentContainer(m)
        for i in m._modules.keys():
            if i == 'shortcut':
                continue
            #print(i)
            res = traverse(child, m._modules[i], i, actions)
            if res == None:
                return None
        if classname not in ['ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model']:
            parent.add_module(m_name, child)
        else:
            return child
    else:
        if classname == 'Linear':
            inp = inp.view(inp.size(0), -1)
            #print(inp.size(1))
            if inp.size(1) > LINEAR_THRESHOLD:
                return None
        if m.fixed or actions[a]:
            m = resizeToFit(Layer(m), inp).cuda()
            inp = m(inp)
            parent.add_module(m_name, m)
        a += 1
    return True

def fixLayers(m):
    layers = flattenModule(m)
    # Initialize
    for l in layers:
        l.fixed = False
    
    layers[-1].fixed=True
    # Fix any layers you want here
    # ----
    #     Fix all shortcut layers and corresponding stride layers, but not pre layers
    for l in layers:
        # Fix final linear and average pooling layer
        cn = l.__class__.__name__
        if hasattr(l, 'stride') and l.stride != (1, 1) and cn == 'Conv2d':
            l.fixed = True
        if cn == 'Linear' or cn == 'AvgPool2d':
            l.fixed = True
    # ----

'''
    Build child model
'''
def build_child_model(m, actions):
    
    # What we want to do here is:
    # Automatically construct containers based on actions of the child
    # We also want to have universality across models
    # Need to handle conv to fc transition
    # In VGG FC is in Sequential called features, in ResNet FC is in Sequential called fc
    # Have a switch that looks out for layers called fc or features
    # Flatten inp on seeing that
    # Need to also incorporate filter channels resizeToFit
    
    actions[0] = 1
    global a
    global inp
    a = 0
    
    inp = Variable(datasetInputTensor.clone()).cuda()
    fixLayers(m)
    # Here we traverse the teacher model, which has a heirarchical structure to generate a child model
    newModel = traverse(None, m, None, actions)
    if newModel == None:
        return None
    resetModel(newModel)
    # Check if any compression has been achieved
    if numParams(newModel) >= parentSize:
        return None
    
    return newModel


def rolloutActions(layers):
    global controller
    hn = Variable(torch.zeros(num_layers * 2, 1, num_hidden))
    cn = Variable(torch.zeros(num_layers * 2 , 1, num_hidden))
    input = Variable(torch.Tensor(len(layers), 1, num_input))
    for i in range(len(layers)):
        input[i] = Layer(layers[i]).toTorchTensor(skipSupport=skipSupport)
    actions, values = controller(input, (hn, cn))
    return actions, values 


def rollout(model_, i):
    global b
    global R_sum
    layers = layersFromModule(model_)
    actions = rolloutActions(layers)
    #fixLayers(model_)
    newModel = build_child_model(model_, [a.data.numpy()[0] for a in actions])
    hashcode = hash(str(newModel)) if newModel else 0
    if hashcode in previousModels and constrained == False:
        R = previousModels[hashcode]
    elif newModel is None:
        R = -1
    else:
        print(newModel)
        #if numParams(newModel) >= 1700000:
        #    return (-1, actions, newModel)
        acc = trainTeacherStudent(model, newModel, dataset, epochs=5)
        R = Reward(acc, numParams(newModel), baseline_acc, parentSize, iter=int(i), constrained=constrained, vars=[numParams(newModel)], cons=[1700000])
        previousModels[hashcode] = R
        # TODO: Turn constrained off after 20 or so iterations
        #C = 1 - float(numParams(newModel))/parentSize
        #R = -1 if acc < 0.88 or C < 0.5 else R
        rewardsPerModel[i] = R
        accsPerModel[i] = acc
        paramsPerModel[i] = numParams(newModel)
        torch.save(newModel, modelSavePath + '%f.net' % i)
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
    valueSeqs = []
    studentModels = []
    for i in range(N):
        model_ = copy.deepcopy(model)
        layers = layersFromModule(model_)
        actions, values = rolloutActions(layers)
        valueSeqs.append(values)
        actionSeqs.append(actions)
        newModel = build_child_model(model_, [a.data.numpy()[0] for a in actions])
        hashcode = hash(str(newModel)) if newModel else 0
        if hashcode in previousModels and constrained == False:
            Rs[i] = previousModels[hashcode]
        elif newModel is None:
            Rs[i] = -1
        else:
            print(newModel)
            torch.save(newModel, modelSavePath + '%f_%f.net' % (e, i))
            newModels.append(newModel)
            studentModels.append(newModel)
            idxs.append(i)
    accs = trainTeacherStudentParallel(model, studentModels, dataset, epochs=5)
    for acc in accs:
        print('Val accuracy: %f' % acc)
    for i in range(len(newModels)):
        print('Compression: %f' % (1.0 - (float(numParams(newModels[i]))/parentSize)))
    R = [Reward(accs[i], numParams(newModels[i]), baseline_acc, parentSize, iter=int(i),     constrained=constrained, vars=[numParams(newModels[i])], cons=[1700000]) for i in range(len(accs))]
    for i in range(len(idxs)):
        Rs[idxs[i]] = R[i]
    for i in range(len(Rs)):
        print('Reward achieved %f' % Rs[i])
    return (Rs, actionSeqs, valueSeqs, newModels)


def rollouts(N, model, e):
    Rs = []
    actionSeqs = []
    models = []
    (Rs, actionSeqs, valueSeqs, models) = rollout_batch(copy.deepcopy(model), N, e)
    return (Rs, actionSeqs, valueSeqs, models)


def update_controller(actionSeqs, valueSeqs, avgR):
    print('Reinforcing for epoch %d' % e)
    LossFn = nn.SmoothL1Loss()
    value_loss = 0
    for (actions, values) in zip(actionSeqs, valueSeqs):
        actions.reinforce(-(values.data-avgR))
        rew = Variable(torch.Tensor([avgR]*values.size(0))).detach()
        value_loss += LossFn(values, rew)
    opti.zero_grad()
    autograd.backward([value_loss] + actionSeqs, [torch.ones(1)]+[None for _ in actionSeqs])
    opti.step()

epochs = 100 
N = 5
prevRs = [0, 0, 0, 0, 0]
for e in range(epochs):
    # Compute N rollouts
    (Rs, actionSeqs, valueSeqs, models) = rollouts(N, model, e)
    # Compute average reward
    avgR = np.mean(Rs)
    print('Average reward: %f' % avgR)
    #b = np.mean(prevRs[-5:])
    prevRs.append(avgR)
    b = R_sum/float(e+1)
    R_sum = R_sum + avgR
    # Update controller
    update_controller(actionSeqs, valueSeqs, avgR)

torch.save(controller, controllerSavePath)
resultsFile = open(modelSavePath + 'results.txt', "w")
output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel)


