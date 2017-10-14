import os
import torch
from torch import nn 
from torch import optim
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import argparse
from rl import *
from architecture import *
import os
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='N2N: Network to Network Compression using Policy Gradient Reinforcement Learning')
parser.add_argument('mode', type=str, choices=['removal', 'shrinkage'],
                    help='Which mode to run the program')
parser.add_argument('dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_old', 'cifar100', 'svhn', 'caltech256'],
                    help='Name of dataset')
parser.add_argument('teacherModel', type=str,
                    help='Path to teacher model')
parser.add_argument('--model', type=str, required=False,
                    help='Path to base model architecture if different from teacherModel')
parser.add_argument('--cuda', type=bool, required=False, default=True,
                    help='Use GPU or not')
parser.add_argument('--gpuids', type=list, required=False, default=[0],
                    help='Which GPUs to use')
parser.add_argument('--debug', type=bool, required=False, default=False,
                    help='Debug mode')
parser.add_argument('--size_constraint', type=int, required=False,
                    help='Add a constraint on size in # parameters')
parser.add_argument('--acc_constraint', type=float, required=False,
                    help='Add a constraint on accuracy in [0, 1]')
parser.add_argument('--controller', type=str, required=False,
                    help='Path to a previously trained controller')
args = parser.parse_args()

if len(args.gpuids) > 1:
    print('Parallel version not implemented yet')
else:
    torch.cuda.set_device(args.gpuids[0])

# ----DATASETS----
if args.dataset == 'mnist':
    import datasets.mnist as dataset
elif args.dataset == 'cifar10':
    import datasets.cifar10 as dataset
elif args.dataset == 'cifar10_old':
    import datasets.cifar10_old as dataset
elif args.dataset == 'cifar100':
    import datasets.cifar100 as dataset
elif args.dataset == 'svhn':
    import datasets.svhn as dataset
elif args.dataset == 'caltech256':
    import datasets.caltech256 as dataset
elif args.dataset == 'imagenet':
    import datasets.imagenet as dataset
else:
    print('Dataset not found: ' + args.dataset)
    quit()

print('Using %s as dataset' % args.dataset)
dataset.cuda = args.cuda
datasetInputTensor = dataset.test_loader.dataset[0][0].unsqueeze(0)
print(datasetInputTensor.size())
baseline_acc = None

# ----MODELS----
# Load teacherModel
teacherModel = torch.load(args.teacherModel)
# Load baseModel (if available)
model = torch.load(args.model) if args.model else deepcopy(teacherModel)

# ----PATHS----
# Define save paths
controllerSavePath = './controllers_%s/' % args.dataset
if not os.path.exists(controllerSavePath):
    os.mkdir(controllerSavePath)
modelSavePath = './models_%s' % args.dataset

# ----HYPERPARAMETERS----
# Initialize controller based on mode
skipSupport = False
num_layers = 2
num_hidden = 30
num_input = 7 if skipSupport else 5
lookup = [0.25 , .5, .5, .5, .5, .5, .6, .7, .8, .9, 1.] # Used for shrinkage only
controller = None
optim_controller = None
lr = 0.003

# ----MODE----
if args.mode == 'removal':
    num_output = 2
    #from controllers.ActorCriticLSTM import *
    from controllers.LSTM import * 
    controllerClass = LSTM
    extraControllerParams = {'bidirectional': True}
    lr = 0.003
elif args.mode == 'shrinkage':
    num_output = len(lookup)
    from controllers.AutoregressiveParam import *
    controllerClass = LSTMAutoParams
    extraControllerParams = {'lookup': lookup}
    lr = 0.1
else:
    print('Mode not known: ' + args.mode)
    quit()


# ----CONSTRAINTS----
size_constraint = args.size_constraint
acc_constraint = args.acc_constraint

# Identify baseline accuracy of base model
dataset.net = model.cuda() if args.cuda else model
print('Testing parent model to determine baseline accuracy')
baseline_acc = baseline_acc if baseline_acc != None else dataset.test()


# Store statistics for each model
previousModels = {}
accsPerModel = {}
paramsPerModel = {}
rewardsPerModel = {}
numSavedModels = 0

# Reward terms for reinforce baseline
R_sum = 0
b = 0

epochs = 100
N = 5 
prevRs = [0] * N
if args.controller:
    controllerClass = args.controller
controller = Controller(controllerClass, num_input, num_output, num_hidden, num_layers, lr=lr, skipSupport=skipSupport, kwargs=extraControllerParams)
architecture = Architecture(args.mode, model, datasetInputTensor, args.dataset, baseline_acc=baseline_acc, lookup=lookup)
# ----MAIN LOOP----
for e in range(epochs):
    # Compute N rollouts
    (Rs, actionSeqs, models) = rollouts(N, model, controller, architecture, dataset, e, size_constraint=size_constraint, acc_constraint=acc_constraint)
    saveModels(e, models, modelSavePath)
    # Compute average reward
    avgR = np.mean(Rs)
    print('Average reward: %f' % avgR)
    #b = np.mean(prevRs[-5:])
    prevRs.append(avgR)
    b = R_sum/float(e+1)
    R_sum = R_sum + avgR
    # Update controller
    print('Reinforcing for epoch %d' % e)
    controller.update_controller(avgR, b)

torch.save(controller, controllerSavePath)
resultsFile = open(os.path.join(modelSavePath, 'results.txt'), "w")
output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel)
