import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms

from Model import Model
from model.resnet import *
from model.lenet import *

def resizeLayer(layer, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1):
    if dilation == 1 and hasattr(layer, 'dilation'):
        dilation = layer.dilation
    if layer.__class__.__name__ is 'Conv2d':
        kernel_size = (kernel_size, kernel_size) if type(kernel_size) is not tuple else kernel_size
        stride = (stride, stride) if type(stride) is not tuple else stride
        padding = (padding, padding) if type(padding) is not tuple else padding
        sd = layer.state_dict()
        sd['weight'].resize_(out_channels, in_channels, kernel_size[0], kernel_size[1])
        if 'bias' in sd:
            sd['bias'].resize_(out_channels)
            # Define new layer
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        else:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        layer.load_state_dict(sd)
    if layer.__class__.__name__ is 'MaxPool2d':
        layer = nn.MaxPool2d(kernel_size, stride=stride, dilation=dilation)
    if layer.__class__.__name__ is 'Linear':
        sd = layer.state_dict()
        sd['weight'].resize_(out_channels, in_channels)
        sd['bias'].resize_(out_channels)
        layer = nn.Linear(in_channels, out_channels)
        layer.load_state_dict(sd)
    if layer.__class__.__name__ is 'ReLU':
        layer = nn.ReLU(inplace=False)
    if layer.__class__.__name__ is 'BatchNorm2d':
        sd = layer.state_dict()
        for k in sd:
            sd[k].resize_(in_channels)
        layer = nn.BatchNorm2d(in_channels, eps=layer.eps, momentum=layer.momentum, affine=layer.affine)
        layer.load_state_dict(sd)
    return layer


def determine_fc_size(inp, model):
    output = model.features(inp)
    return output.view(-1).size()[0]

def output_results(resultsFile, accsPerModel, paramsPerModel, rewardsPerModel):
    resultsString = ''
    s = '-- Models ranked by accuracy --'
    print(s)
    resultsString += s + "\n"
    i = 1
    for k in sorted(accsPerModel, key=accsPerModel.get)[::-1]:
        s = '#%d: model%f acc %f' % (i, k, accsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    i = 1
    s = '-- Models ranked by size --'
    print(s)
    resultsString += s + "\n"
    for k in sorted(paramsPerModel, key=paramsPerModel.get):
        s = '#%d: model%f size %d' % (i, k, paramsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    i = 1
    for k in sorted(rewardsPerModel, key=rewardsPerModel.get)[::-1]:
        s = '#%d: model%f reward %f ' % (i, k, rewardsPerModel[k])
        print(s)
        resultsString += s + "\n"
        i += 1
    if resultsFile:
        resultsFile.write(resultsString)

def numParams(model):
        return sum([len(w.view(-1)) for w in model.parameters()])


def train(dataset, net):
    net.add_module('LogSoftmax', nn.LogSoftmax())
    print (dataset.args.cuda)
    dataset.net = net.cuda() if dataset.args.cuda else net.cpu()
    train_acc = []
    val_acc = [-1]
    for i in xrange(1, dataset.args.epochs+1):
        train_acc.append(dataset.train(i))
        acc = dataset.test()
        if i >= 2 and acc < 0.2:
            break
        print('Val acc: ' + str(acc))
        val_acc.append(acc)
    return max(val_acc)

def removeLayers(m, type):
    if m.__class__.__name__ == type:
        return True
    for k in m._modules.keys():
        res = removeLayers(m._modules[k], type)
        if res:
            del m._modules[k]
    return False

import time
import itertools
def trainTeacherStudent(teacher, student, dataset, epochs=5, lr=0.0005):
    startTime = time.time()
    student = student.cuda()
    teacher = teacher.cuda()
    # If there is a log softmax somewhere, delete it in both teacher and student
    removeLayers(teacher, type='LogSoftmax')
    removeLayers(teacher, type='Softmax')
    removeLayers(student, type='LogSoftmax')
    removeLayers(student, type='Softmax')
    MSEloss = nn.MSELoss().cuda()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    student.train()
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(dataset.train_loader):
            data = data.cuda()
            data = Variable(data)
            optimizer.zero_grad()
            studentOutput = student(data)
            teacherOutput = teacher(data).detach()
            loss = MSEloss(studentOutput, teacherOutput)
            loss.backward()
            optimizer.step()
        student.add_module('LogSoftmax', nn.LogSoftmax())
        dataset.net = student
        removeLayers(student, type='LogSoftmax')
        print(dataset.test())
        print('Train Epoch: {} \tLoss: {:.6f}'.format(i, loss.data[0]))
    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataset.net = student
    acc = dataset.test()
    print('Time elapsed: {}'.format(time.time()-startTime))
    return acc 

import torch.nn.functional as F
def trainTeacherStudentRand(teacher, student, dataset, epochs=50, lr=0.0001):
    startTime = time.time()
    student = student.cuda()
    teacher = teacher.cuda()
    # If there is a log softmax somewhere, delete it in both teacher and student
    removeLayers(teacher, type='LogSoftmax')
    removeLayers(teacher, type='Softmax')
    removeLayers(student, type='LogSoftmax')
    removeLayers(student, type='Softmax')
    MSEloss = nn.MSELoss().cuda()
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=5e-4)
    student.train()
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(dataset.train_loader):
            data = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(torch.rand(64, 3, 32, 32)).cuda()
            data = Variable(data)
            optimizer.zero_grad()
            studentOutput = student(data)
            teacherOutput = teacher(data).detach()
            loss = MSEloss(studentOutput, teacherOutput)
            loss.backward()
            optimizer.step()
        student.add_module('LogSoftmax', nn.LogSoftmax())
        dataset.net = student
        removeLayers(student, type='LogSoftmax')
        print(dataset.test())
        print('Train Epoch: {} \tLoss: {:.6f}'.format(i, loss.data[0]))
    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataset.net = student
    acc = dataset.test()
    print('Time elapsed: {}'.format(time.time()-startTime))
    return acc

def trainTeacherStudentNew(teacher, student, dataset, epochs=5, lr=0.0005, T=3.0, lambd=0.3):
    startTime = time.time()
    student = student.cuda()
    teacher = teacher.cuda()
    # If there is a log softmax somewhere, delete it in both teacher and student
    removeLayers(teacher, type='LogSoftmax')
    removeLayers(teacher, type='Softmax')
    removeLayers(student, type='LogSoftmax')
    removeLayers(student, type='Softmax')
    MSEloss = nn.MSELoss().cuda()
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=5e-4)
    student.train()
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(dataset.train_loader):
            data = data.cuda()
            data = Variable(data)
            targets = targets.cuda()
            targets = Variable(targets)
            optimizer.zero_grad()
            studentOutput = F.log_softmax(student(data)/T)
            teacherOutput = F.log_softmax(teacher(data).detach()/T)
            loss = (1-lambd)*MSEloss(studentOutput, teacherOutput) + lambd*F.nll_loss(studentOutput, targets)
            loss.backward()
            optimizer.step()
        student.add_module('LogSoftmax', nn.LogSoftmax())
        dataset.net = student
        removeLayers(student, type='LogSoftmax')
        print(dataset.test())
        print('Train Epoch: {} \tLoss: {:.6f}'.format(i, loss.data[0]))
    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataset.net = student
    acc = dataset.test()
    print('Time elapsed: {}'.format(time.time()-startTime))
    return acc

def trainTeacherStudentParallel(teacher, students, dataset, epochs=5, lr=0.0005):
    if len(students) == 0:
        return []
    startTime = time.time()
    students = [student.cuda() for student in students]
    teacher = teacher.cuda()
    # If there is a log softmax somewhere, delete it in both teacher and student
    removeLayers(teacher, type='LogSoftmax')
    for student in students:
        removeLayers(student, type='LogSoftmax')
        student.train()
    MSEloss = nn.MSELoss().cuda()
    optimizers = [optim.Adam(student.parameters(), lr=lr, weight_decay=5e-4) for student in students]
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(dataset.train_loader):
            data = data.cuda()
            teacherOutput = teacher(Variable(data)).detach()
            for j in range(len(students)):
                studentData = Variable(data)
                optimizers[j].zero_grad()
                studentOutput = students[j](studentData)
                loss = MSEloss(studentOutput, teacherOutput)
                loss.backward()
                optimizers[j].step()
        print('Train Epoch: {}'.format(i))
        for j in range(len(students)):
            removeLayers(students[j], type='LogSoftmax')
            students[j].add_module('LogSoftmax', nn.LogSoftmax())
            dataset.net = students[j]
            print('Student {} acc {}'.format(j, dataset.test()))
            removeLayers(student, type='LogSoftmax')
        
    accs = []
    for student in students:
        removeLayers(student, type='LogSoftmax')
        student.add_module('LogSoftmax', nn.LogSoftmax())
        dataset.net = student
        accs.append(dataset.test())
    print('Time elapsed {}'.format(time.time() - startTime))
    return accs

def trainNormal(studentModel, dataset, epochs=5):
    return trainNormalParallel([studentModel], dataset, epochs)[0]

def trainNormalParallel(studentModels, dataset, epochs=5):
    accs = []
    for model in studentModels:
        dataset.net = model
        for i in range(1, epochs+1):
            dataset.train(i)
        acc = dataset.test()
        accs.append(acc)
    return accs


layerTypes = ['Unknown', 'Conv2d', 'MaxPool2d', 'ReLU', 'BatchNorm2d', 'Linear', 'Dropout', 'LogSoftmax', 'AvgPool2d', 'L2Norm', 'Softmax']
def getLayerType(layer):
    name = layer.__class__.__name__
    return max(layerTypes.index(name), 0)

import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)


def resetModel(m):
    if len(m._modules) == 0 and hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        return
    for i in m._modules.values():
        resetModel(i)

'''
def resetModel(model):
    for l in model.features._modules.values():
        if hasattr(l, 'reset_parameters'):
            l.reset_parameters()
    
    for l in model.classifier._modules.values():
        if hasattr(l, 'reset_parameters'):
            l.reset_parameters()
    #model.apply(weights_init)
    return model
'''

import Layer
def resizeToFit(layer, inp):
    if layer._layer.__class__.__name__ is 'Linear':
        in_channels = inp.view(inp.size(0), -1).size(1)
        return resizeLayer(layer._layer, in_channels, layer._layer.out_features)
    in_channels = inp.size(1)
    if 'weight' in layer._layer._parameters:
        _, kernel_size, stride, out_channels, padding = layer.getRepresentation()
        return resizeLayer(layer._layer, in_channels, out_channels, kernel_size, stride, padding)
    if layer._layer.__class__.__name__ is 'ReLU':
        return nn.ReLU(inplace=False)
    return layer._layer

def createParentContainer(m):
    classname = m.__class__.__name__
    if classname == 'Sequential':
        return nn.Sequential()
    elif classname in ['BasicBlock', 'Bottleneck', 'BasicBlockModifiable']:
        return BasicBlockModifiable(shortcut=m.shortcut if hasattr(m, 'shortcut') else None)
    elif classname == 'ResNet' or classname == 'ResNetModifiable':
        return ResNetModifiable()
    elif classname == 'VGG':
        return Model(None, None)
    elif classname == 'LeNet':
        return Model(None, None)
    elif classname == 'mnist_model':
        return Model(None, None)
    elif classname == 'Model':
        return Model(None, None)
    elif classname == 'SSD':
        from model.ssd import SSDModifiable
        return SSDModifiable()
    elif classname == 'ModuleList':
        return nn.ModuleList()


def flattenModule(m):
    if len(m._modules) == 0:
        return [m]
    top = []
    for i in m._modules.values():
        bottom = flattenModule(i)
        top.extend(bottom)
    return top


def layersFromModule(m):
    if len(m._modules) == 0:
        m.skipstart = 0
        m.skipend = 0
        return [m]
    top = []
    for i in m._modules:
        bottom = layersFromModule(m._modules[i])
        #print(i, bottom)
        if i in ['layers']:
            # Introduce skip connections to layers in bottom
            n = len(bottom) 
            for j in range(n):
                bottom[j].skipstart = j
                bottom[j].skipend = n - j - 1
        top.extend(bottom)
    return top
