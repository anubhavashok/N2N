import torch
from torch import nn
from torch.nn import LSTMCell
from torch.autograd import Variable

class LSTMAuto(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LSTMAuto, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # Output of previous iteration appended to input
        self.lstmCell = LSTMCell(output_size + input_size, hidden_size)
        # Softmax variables
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
    
    def forward(self, input, hx):
        outputs = []
        output = torch.Tensor(1, self.output_size)
        # Keep layer
        output[0][1] = 1
        output[0][0] = 0
        output = Variable(output)
        hn, cn = hx
        for i in range(len(input)):
            input_augmented = Variable(torch.cat([output.data, input[i].data], 1))
            hn, cn = self.lstmCell(input_augmented, (hn, cn))
            output = self.softmax(self.linear(hn))
            outputs.append(output)
        return torch.stack(outputs)

'''
from Autoregressive import *
from torch.autograd import *
inp = Variable(torch.rand(21, 1, 5))
hx = (Variable(torch.Tensor(1, 10)), Variable(torch.Tensor(1, 10)))

lstm = LSTMAuto(5, 2, 10, 21)
lstm(inp, hx)
'''
