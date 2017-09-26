'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential()
        self.classifier = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(1, 6, 5))
        self.features.add_module('relu1', nn.ReLU())
        self.features.add_module('pool1', nn.MaxPool2d(2, 2))
        self.features.add_module('conv2', nn.Conv2d(6, 16, 5))
        self.features.add_module('relu2', nn.ReLU())
        self.features.add_module('pool2', nn.MaxPool2d(2, 2))
        self.classifier.add_module('linear1', nn.Linear(16*4*4, 120))
        self.classifier.add_module('relu1', nn.ReLU())
        self.classifier.add_module('linear2', nn.Linear(120, 84))
        self.classifier.add_module('relu2', nn.ReLU())
        self.classifier.add_module('linear3', nn.Linear(84, 10))
        self.classifier.add_module('lsm', nn.LogSoftmax())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
