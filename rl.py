import torch
import copy 
from Layer import * 
from utils import *
from architecture import *
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import autograd



class Controller:
    def __init__(self, controller, lr=0.003, skipSupport=False):
        self.controller = controller
        self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.skipSupport = skipSupport

    def update_controller(self, actionSeqs, avgR):
        print('Reinforcing for epoch %d' %e)
        for actions in actionSeqs:
            if isinstance(actions, list):
                for action in actions:
                    action.reinforce(avgR - b)
            else:
                actions.reinforce(avgR - b)
            self.optimizer.zero_grad()
            autograd.backward(actions, [None for _ in actions])
        self.optimizer.step()

    def rolloutActions(self, layers):
        num_input  = self.controller.lstm.input_size
        num_hidden = self.controller.lstm.hidden_size
        num_layers = self.controller.lstm.num_layers
        num_directions = 2 if self.controller.lstm.bidirectional else 1
        hn = Variable(torch.zeros(num_layers * num_directions, 1, num_hidden))
        cn = Variable(torch.zeros(num_layers * num_directions, 1, num_hidden))
        input = Variable(torch.Tensor(len(layers), 1, num_input))
        for i in range(len(layers)):
            input[i] = Layer(layers[i]).toTorchTensor(skipSupport=self.skipSupport)
        actions = self.controller(input, (hn, cn))
        return actions


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

def Reward(acc, params, baseline_acc, baseline_params):
    R_a = (acc/baseline_acc) #if acc > 0.92 else -1
    C = (float(baseline_params - params))/baseline_params
    R_c = C*(2-C)
    if constrained:
        return getConstrainedReward(R_a, R_c, cons, vars, iter)
    return (R_a) * (R_c)

previousModels = {}
def rollout_batch(model, controller, architecture, dataset, N, e):
    newModels = []
    idxs = []
    Rs = [0]*N
    actionSeqs = []
    studentModels = []
    for i in range(N):
        model_ = copy.deepcopy(model)
        layers = layersFromModule(model_)
        actions = controller.rolloutActions(layers)
        actionSeqs.append(actions)
        newModel = architecture.generateChildModel([a.data.numpy()[0] for a in actions])
        hashcode = hash(str(newModel)) if newModel else 0
        if hashcode in previousModels and constrained == False:
            Rs[i] = previousModels[hashcode]
        elif newModel is None:
            Rs[i] = -1
        else:
            print(newModel)
            #torch.save(newModel, modelSavePath + '%f_%f.net' % (e, i))
            newModels.append(newModel)
            studentModels.append(newModel)
            idxs.append(i)
    accs = trainNormalParallel(studentModels, dataset, epochs=5) if architecture.datasetName is 'caltech256' else trainTeacherStudentParallel(model, studentModels, dataset, epochs=5)
    for acc in accs:
        print('Val accuracy: %f' % acc)
    for i in range(len(newModels)):
        print('Compression: %f' % (1.0 - (float(numParams(newModels[i]))/architecture.parentSize)))
    R = [Reward(accs[i], numParams(newModels[i]), architecture.baseline_acc, architecture.parentSize, iter=int(e), constrained=constrained, vars=[numParams(newModels[i])], cons=[1700000]) for i in range(len(accs))]
    for i in range(len(idxs)):
        Rs[idxs[i]] = R[i]
    for i in range(len(Rs)):
        print('Reward achieved %f' % Rs[i])
    return (Rs, actionSeqs, newModels)


def rollouts(N, model, controller, architecture, dataset, e):
    Rs = []
    actionSeqs = []
    models = []
    (Rs, actionSeqs, models) = rollout_batch(copy.deepcopy(model), controller, architecture, dataset, N, e)
    return (Rs, actionSeqs, models)

