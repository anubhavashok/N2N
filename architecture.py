import torch
from utils import *
from Layer import *
from Model import *
import copy

class Architecture:
    def __init__(self, model, datasetInputTensor, datasetName, LINEAR_THRESHOLD=50000, baseline_acc=.5):
        self.model = model
        self.datasetInputTensor = datasetInputTensor
        self.a  = 0
        self.inp = None
        self.LINEAR_THRESHOLD = LINEAR_THRESHOLD
        self.datasetName = datasetName
        self.parentSize = numParams(model)
        self.baseline_acc = baseline_acc 

    def traverse(self, parent, m, m_name, actions):
        classname = m.__class__.__name__
        if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model']:
            child = createParentContainer(m)
            for i in m._modules.keys():
                if i == 'shortcut':
                    continue
                res = self.traverse(child, m._modules[i], i, actions)
                if res == None:
                    return None
            if classname not in ['ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model']:
                parent.add_module(m_name, child)
            else:
                return child
        else:
            if classname == 'Linear':
                self.inp = self.inp.view(self.inp.size(0), -1)
                if self.inp.size(1) > self.LINEAR_THRESHOLD:
                    return None
            if m.fixed or actions[self.a]:
                m = resizeToFit(Layer(m), self.inp).cuda()
                self.inp = m(self.inp)
                parent.add_module(m_name, m)
            self.a += 1
        return True


    def fixLayers(self, m):
        # TODO: Make this function generalize to most models
        # We basically want to make sure at least one fc layer exists
        # We also want to make sure that the stride layer for downsampling does not get removed
        layers = flattenModule(m)
        # Initialize
        for l in layers:
            l.fixed = False
        layers[-1].fixed = True
        for l in layers:
            cn = l.__class__.__name__
            if hasattr(l, 'stride') and l.stride != (1, 1) and cn == 'Conv2d':
                l.fixed = True
            if cn == 'Linear' or cn == 'AvgPool2d':
                l.fixed = True

    def generateChildModel(self, actions):
        m = copy.deepcopy(self.model)
        actions[0] = 1
        self.a = 0
        self.inp = Variable(self.datasetInputTensor.clone()).cuda()
        self.fixLayers(m)
        newModel = self.traverse(None, m, None, actions)
        if newModel == None or numParams(newModel) >= self.parentSize:
            return None
        resetModel(newModel)
        return newModel
