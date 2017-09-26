import torch
from torch import nn
from torch.nn import LSTMCell
from torch.autograd import Variable

class LSTMAuto(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTMAuto, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Output of previous iteration appended to input
        self.layers = []
        input_size += 1
        for i in range(num_layers):
            self.layers.append(LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # Softmax variables
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forwardLayer(self, input, hn, cn, layer):
        return (hn, cn)
    
    def forwardLayers(self, input, hns, cns, layers):
        new_hns = []
        new_cns = []
        (hn, cn) = layers[0](input, (hns[0], cns[0]))
        new_hns.append(hn)
        new_cns.append(cn)
        #hns[0], cns[0] = hn, cn
        for i in range(1, len(layers)):
            (hn, cn) = layers[i](hn, (hns[i], cns[i]))
            new_hns.append(hn)
            new_cns.append(cn)
            #hns[i] = hn
            #cns[i] = cn
        return hn, (new_hns, new_cns)
    
    def forward(self, input, hx):
        outputs = []
        output = torch.Tensor(1, 1)
        # Keep layer
        output[0][0] = 1
        output = Variable(output)
        hns, cns = hx
        for i in range(len(input)):
            input_augmented = Variable(torch.cat([output.data.float(), input[i].data], 1))
            output, (hns, cns) = self.forwardLayers(input_augmented, hns, cns, self.layers)
            probs = self.softmax(self.linear(output))
            output = probs.multinomial()
            outputs.append(output)
        return outputs

'''
from Autoregressive import *
from torch.autograd import *
inp = Variable(torch.rand(21, 1, 5))
hx = ([Variable(torch.Tensor(1, 10))]*num_layers, [Variable(torch.Tensor(1, 10))]*num_layers)

lstm = LSTMAuto(5, 2, 10, 2)
lstm(inp, hx)
'''
