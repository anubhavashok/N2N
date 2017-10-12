import torch
from utils import *
from Layer import *
from Model import *
import copy
import numpy as np


LINEAR_THRESHOLD = 50000

def applyActionsShrinkage(m, action, inp, lookup):
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


class Architecture:
    def __init__(self, mode, model, datasetInputTensor, datasetName, LINEAR_THRESHOLD=50000, baseline_acc=.5, lookup=None):
        self.mode = mode
        self.model = model
        self.datasetInputTensor = datasetInputTensor
        self.a  = 0
        self.inp = None
        self.LINEAR_THRESHOLD = LINEAR_THRESHOLD
        self.datasetName = datasetName
        self.parentSize = numParams(model)
        self.baseline_acc = baseline_acc 
        self.lookup = lookup

    def traverse_removal(self, parent, m, m_name, actions):
        classname = m.__class__.__name__
        if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model', 'ResNetModifiable', 'BasicBlockModifiable']:
            child = createParentContainer(m)
            for i in m._modules.keys():
                if i == 'shortcut':
                    continue
                res = self.traverse_removal(child, m._modules[i], i, actions)
                if res == None:
                    return None
            if classname not in ['ResNet', 'VGG', 'LeNet', 'mnist_model', 'Model', 'ResNetModifiable']:
                parent.add_module(m_name, child)
            else:
                return child
        else:
            # childless layers -> we can shrink/remove these
            if classname == 'Linear':
                self.inp = self.inp.view(self.inp.size(0), -1)
                if self.inp.size(1) > self.LINEAR_THRESHOLD:
                    return None
            # perform removal/shrinkage
            # if self.mode == 'removal':
            #     add if ok
            # else if self.mode == 'shrinkage':
            #     shrink layer
            #     add if ok
            # else
            if m.fixed or actions[self.a]:
                m = resizeToFit(Layer(m), self.inp).cuda()
                self.inp = m(self.inp)
                parent.add_module(m_name, m)
            self.a += 1
        return True

    def traverse_shrinkage(self, parent, m, m_name, actions):
        classname = m.__class__.__name__
        if classname in ['Sequential', 'BasicBlock', 'Bottleneck', 'ResNet', 'VGG', 'LeNet', 'Model', 'ResNetModifiable', 'BasicBlockModifiable']:
            # Change the number of input channels of the first conv of the shortcut layer
            oldInp = Variable(copy.deepcopy(self.inp.data))
            child = createParentContainer(m)
            if classname in ['BasicBlock', 'BottleNeck', 'BasicBlockModifiable']:
                self.fixBlockLayers(m)
                m = self.processBlock(actions, m, self.lookup, self.inp.size(1)).cuda()
                self.inp = m.layers(self.inp.cuda())
                child = m
            else:
                for i in m._modules.keys():
                    res = self.traverse_shrinkage(child, m._modules[i], i, actions)
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
                self.inp = self.inp.view(self.inp.size(0), -1)
                #print(inp.size(1))
                if self.inp.size(1) > LINEAR_THRESHOLD or self.inp.size(1) < 10:
                    print('Linear layer too large')
                    return None
            action = actions[self.a][:]
            m = applyActionsShrinkage(m, action, self.inp, self.lookup)
            if m == None:
                return None
            try:
                self.inp = m.cuda()(self.inp)
            except:
                print('Error in model, probably because of receptive field size')
                return None
            parent.add_module(m_name, m)
            self.a += 1
        return True

    def processBlock(self, actions, m, lookup, input_size):
        finalAction = actions[self.a+len(m.layers._modules)-1][3]
        finalActionUsed = False

        secondFinalAction = actions[self.a+len(m.layers._modules)-2][3]
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
                o = max(int(o * self.lookup[finalAction]), 10)
                finalActionUsed = True
            elif hasShortcut:
                o = max(int(o * self.lookup[finalAction]), 10)
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
                o = max(int(o * self.lookup[secondFinalAction]), 10)
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
        for _ in range(len(m.layers._modules)-2):
        #    actions[a].detach()
            self.a += 1

        #if not secondFinalActionUsed:
        #    actions[a].detach()
        self.a += 1

        #if not finalActionUsed:
        #    actions[a].detach()
        self.a += 1
        return m


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

    def fixBlockLayers(self, m):
        # Only allow num_filters of conv layers to change
        for mm in m.layers._modules.values():
            mm.fixed = [True]*5
        m.layers._modules.values()[0].fixed = [True, True, True, False, True]
        #m._modules.values()[-2].fixed = [True, True, True, False, True]
    
    def fixParams(self, m):
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
    

    def generateChildModel(self, actions):
        m = copy.deepcopy(self.model)
        self.a = 0
        self.inp = Variable(self.datasetInputTensor.clone()).cuda()
        if self.mode == 'shrinkage':
            # Reshape actions to [Layer, Param]
            actions = np.reshape(actions, (-1, 5))
            self.fixParams(m)
            newModel = self.traverse_shrinkage(None, m, None, actions)
        else:
            actions[0] = 1
            self.fixLayers(m)
            newModel = self.traverse_removal(None, m, None, actions)
        if newModel == None or numParams(newModel) >= self.parentSize:
            return None
        resetModel(newModel)
        return newModel
