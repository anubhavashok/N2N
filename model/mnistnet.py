from torch import nn
class mnist_model(nn.Module):
    
    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.AvgPool2d(6, 6)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.ReLU(True)
        )
    
    def forward(self, inputs):
        out = self.feats(inputs)
        out = out.view(-1, 128)
        out = self.classifier(out)
        return out
