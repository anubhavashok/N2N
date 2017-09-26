import torch.nn as nn

class Model(nn.Module):
    def __init__(self, features, classifier):
        super(Model, self).__init__()
        if features:
            self.features = features
        if classifier:
            self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
    
