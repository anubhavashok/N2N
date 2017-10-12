import torch
from torch import nn
from torch.nn import LSTMCell
from torch.autograd import Variable
import numpy as np

class LSTMAutoParams(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, lookup):
        super(LSTMAutoParams, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookup = lookup
        # Output of previous iteration appended to input
        self.layers = []
        for i in range(num_layers):
            self.layers.append(LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # Softmax variables
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
    
    def forwardLayers(self, input, hns, cns, layers):
        new_hns = []
        new_cns = []
        (hn, cn) = layers[0](input, (hns[0], cns[0]))
        new_hns.append(hn)
        new_cns.append(cn)
        for i in range(1, len(layers)):
            (hn, cn) = layers[i](hn, (hns[i], cns[i]))
            new_hns.append(hn)
            new_cns.append(cn)
        return hn, (new_hns, new_cns)
    
    def forward(self, input, hx):
        actions = []
        output = torch.Tensor(1, self.output_size)
        # Keep layer
        hns, cns = hx
        action = Variable(torch.Tensor(1))
        action[0] = 0
        for i in range(len(input)):
            input_augmented = input[i]
            for j in range(self.input_size):
                # incorporate previous decision into input[i][j]
                output, (hns, cns) = self.forwardLayers(input_augmented, hns, cns, self.layers)
                output = self.softmax(self.linear(output))
                action = output.squeeze(1).multinomial()
                actions.append(action)
                # Update input_augmented for next iteration
                intAction = int(action.data.numpy()) 
                mult = np.ones((self.input_size)).astype(np.float32)
                # Don't change type here
                mult[j] = 1 if j == 0 else self.lookup[intAction]
                input_augmented = Variable(input_augmented.data * torch.from_numpy(mult))
        return actions


'''
import random
from AutoregressiveParam import *
import numpy as np

inp = np.zeros((21, 1, 5))

for i in range(inp.shape[0]):
    r = random.random()
    num = 1/(1 + pow(np.e, r))
    for j in range(inp.shape[2]):
        inp[i][0][j] = num

num_layers = 2

inp = Variable(torch.from_numpy(inp.astype(np.float32)))
hx = ([Variable(torch.Tensor(1, 10))]*num_layers, [Variable(torch.Tensor(1, 10))]*num_layers)

lookup = [1, 0]
lstm = LSTMAutoParams(5, 2, 10, num_layers, lookup)
lstm(inp, hx)
'''
